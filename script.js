/* global THREE, Hands */

const TIP_INDICES = [4, 8, 12, 16, 20];
const PALM_INDEX = 9;
const MAX_POINTS = 14;
const TRAIL_DECAY = 0.9945;
const DENSITY_DECAY = 0.991;
const VELOCITY_DECAY = 0.972;
const FLOW_DECAY = 0.993;

const VISUAL = {
  handRadiusPx: 220,
  splatRadius: 0.032,
  smoothFactor: 0.38,
  trailRetain: 0.52,
  distortScale: 0.028,
  chromaticAberration: 0.0008,
  glassBlend: 0.72,
  glowStrength: 0.04,
};

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const overlay = document.getElementById("overlay");
const hud = document.getElementById("hud");
const startBtn = document.getElementById("startBtn");
const envWarning = document.getElementById("envWarning");

let renderer, scene, camera, quad, videoTexture;
let hands;
let handBusy = false;
let handBusyTimeoutId = null;
let frameCount = 0;
let running = false;
let starting = false;
let clock = null;
let rtType = null;
let simW = 256;
let simH = 256;

const points = Array.from({ length: MAX_POINTS }, () => ({
  x: 0.5, y: 0.5, vx: 0, vy: 0, strength: 0, active: 0,
}));
const fingerUniforms = Array.from({ length: 12 }, () => new THREE.Vector2(0.5, 0.5));
let fingerDisplayCount = 0;
let pointCount = 0;
let prevPositions = new Map();

let velocityPP, densityPP, flowPP, trailPP, pressurePP;
let displayRT, bloomRT, prevFrameRT;
let splatVelMat, splatDenMat, advectVelMat, advectDenMat, flowMat, trailMat;
let pressureMat, divMat, gradSubtractMat, displayMat, bloomMat, blurMat;
let compositeMat, motionBlurMat, copyMat;
let finalRT;

let orthoCam = null;

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function showEnvWarning(message) {
  if (!envWarning) return;
  if (message) {
    envWarning.textContent = message;
    envWarning.hidden = false;
  } else {
    envWarning.hidden = true;
    envWarning.textContent = "";
  }
}

function showFatal(message) {
  console.error(message);
  showEnvWarning(message);
  setStatus(message);
  if (startBtn) {
    startBtn.disabled = false;
    startBtn.textContent = "开启摄像头体验";
  }
  starting = false;
}

function isMobileDevice() {
  return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
}

function checkLibraries() {
  if (typeof THREE === "undefined") {
    showFatal(
      "Three.js 未加载。请检查网络，或在 Edge 中：设置 → 隐私 → 关闭对本站的跟踪防护后刷新。"
    );
    return false;
  }
  if (typeof Hands === "undefined") {
    showFatal("MediaPipe Hands 未加载。请检查网络后刷新页面。");
    return false;
  }
  return true;
}

function getRenderTargetType(rendererInstance) {
  if (!rendererInstance.capabilities.isWebGL2) {
    return THREE.UnsignedByteType;
  }
  try {
    const gl = rendererInstance.getContext();
    if (gl.getExtension("EXT_color_buffer_float") || gl.getExtension("OES_texture_half_float")) {
      return THREE.HalfFloatType;
    }
  } catch (e) {
    /* use byte fallback */
  }
  return THREE.UnsignedByteType;
}

function mirrorX(x) {
  return 1.0 - x;
}

class PingPong {
  constructor(w, h, options = {}) {
    const opts = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
      stencilBuffer: false,
      ...options,
    };
    this.read = new THREE.WebGLRenderTarget(w, h, opts);
    this.write = new THREE.WebGLRenderTarget(w, h, opts);
  }

  swap() {
    const t = this.read;
    this.read = this.write;
    this.write = t;
  }

  clear(renderer, color = 0x000000) {
    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(this.read);
    renderer.setClearColor(color, 0);
    renderer.clear();
    renderer.setRenderTarget(this.write);
    renderer.clear();
    renderer.setRenderTarget(prev);
  }
}

function blit(material, target) {
  if (!renderer || !quad || !orthoCam) return;
  if (material.uniforms?.uResolution) {
    material.uniforms.uResolution.value.set(simW, simH);
  }
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(target);
  quad.material = material;
  renderer.render(scene, orthoCam);
  renderer.setRenderTarget(prev);
}

const FULLSCREEN_VS = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

function initShaders() {
  splatVelMat = new THREE.ShaderMaterial({
    uniforms: {
      uVelocity: { value: null },
      uPoints: { value: new Array(MAX_POINTS).fill(new THREE.Vector4()) },
      uPointCount: { value: 0 },
      uTime: { value: 0 },
      uRadius: { value: VISUAL.splatRadius },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uVelocity;
      uniform vec4 uPoints[14];
      uniform int uPointCount;
      uniform float uTime;
      uniform float uRadius;
      varying vec2 vUv;

      float gauss(float d, float r) {
        return exp(-d * d / (r * r + 0.0001));
      }

      void main() {
        vec2 uv = vUv;
        vec2 vel = texture2D(uVelocity, uv).xy;

        for (int i = 0; i < 14; i++) {
          if (i >= uPointCount) break;
          vec4 p = uPoints[i];
          float power = p.w;
          if (power < 0.02) continue;

          vec2 delta = uv - p.xy;
          float dist = length(delta);
          float r = uRadius * (0.65 + power * 0.5);
          float g = gauss(dist, r);

          vec2 moveForce = p.zw * 12.0;
          vec2 dir = dist > 0.0005 ? delta / dist : vec2(0.0);
          vec2 tangent = vec2(-dir.y, dir.x);
          vec2 swirl = tangent * 0.4 * sin(uTime * 4.5 + float(i) * 0.7);

          vel += (moveForce + swirl) * g * power;
        }

        gl_FragColor = vec4(vel, 0.0, 1.0);
      }
    `,
  });

  splatDenMat = new THREE.ShaderMaterial({
    uniforms: {
      uDensity: { value: null },
      uPoints: { value: new Array(MAX_POINTS).fill(new THREE.Vector4()) },
      uPointCount: { value: 0 },
      uTime: { value: 0 },
      uRadius: { value: VISUAL.splatRadius },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uDensity;
      uniform vec4 uPoints[14];
      uniform int uPointCount;
      uniform float uTime;
      uniform float uRadius;
      varying vec2 vUv;

      float gauss(float d, float r) {
        return exp(-d * d / (r * r + 0.0001));
      }

      void main() {
        vec2 uv = vUv;
        float dens = texture2D(uDensity, uv).r;

        for (int i = 0; i < 14; i++) {
          if (i >= uPointCount) break;
          vec4 p = uPoints[i];
          float power = p.w;
          if (power < 0.02) continue;

          vec2 delta = uv - p.xy;
          float dist = length(delta);
          float r = uRadius * (0.65 + power * 0.5);
          float g = gauss(dist, r);
          float gWide = gauss(dist, r * 2.5);

          dens += g * (0.18 + power * 0.12);
          float ripple = sin(dist * 50.0 - uTime * 8.0) * gWide * 0.14 * power;
          dens += ripple;
          dens += gWide * 0.06 * power;
        }

        gl_FragColor = vec4(dens, dens, dens, 1.0);
      }
    `,
  });

  advectVelMat = new THREE.ShaderMaterial({
    uniforms: {
      uVelocity: { value: null },
      uSource: { value: null },
      uDt: { value: 0.016 },
      uDecay: { value: VELOCITY_DECAY },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uVelocity;
      uniform sampler2D uSource;
      uniform float uDt;
      uniform float uDecay;
      uniform vec2 uResolution;
      varying vec2 vUv;

      void main() {
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vec2 uv = vUv - vel * uDt * 1.6;
        uv = clamp(uv, 0.002, 0.998);
        vec2 outVel = texture2D(uSource, uv).xy * uDecay;
        gl_FragColor = vec4(outVel, 0.0, 1.0);
      }
    `,
  });

  advectDenMat = new THREE.ShaderMaterial({
    uniforms: {
      uVelocity: { value: null },
      uSource: { value: null },
      uDt: { value: 0.016 },
      uDecay: { value: DENSITY_DECAY },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uVelocity;
      uniform sampler2D uSource;
      uniform float uDt;
      uniform float uDecay;
      uniform vec2 uResolution;
      varying vec2 vUv;

      void main() {
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vec2 uv = vUv - vel * uDt * 1.4;
        uv = clamp(uv, 0.002, 0.998);
        float d = texture2D(uSource, uv).r * uDecay;
        gl_FragColor = vec4(d, d, d, 1.0);
      }
    `,
  });

  flowMat = new THREE.ShaderMaterial({
    uniforms: {
      uFlow: { value: null },
      uVelocity: { value: null },
      uDecay: { value: FLOW_DECAY },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uFlow;
      uniform sampler2D uVelocity;
      uniform float uDecay;
      varying vec2 vUv;

      void main() {
        vec2 prevFlow = texture2D(uFlow, vUv).xy * uDecay;
        vec2 vel = texture2D(uVelocity, vUv).xy;
        float speed = length(vel);
        vec2 dir = speed > 0.0001 ? vel / speed : vec2(0.0);
        vec2 flow = prevFlow + dir * speed * 0.55;
        flow = clamp(flow, -2.0, 2.0);
        float mag = length(flow);
        gl_FragColor = vec4(flow, mag, 1.0);
      }
    `,
  });

  trailMat = new THREE.ShaderMaterial({
    uniforms: {
      uTrail: { value: null },
      uDensity: { value: null },
      uDecay: { value: TRAIL_DECAY },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uTrail;
      uniform sampler2D uDensity;
      uniform float uDecay;
      varying vec2 vUv;

      void main() {
        float prev = texture2D(uTrail, vUv).r * uDecay;
        float dens = texture2D(uDensity, vUv).r;
        float trail = max(prev, dens * 0.52);
        gl_FragColor = vec4(trail, trail, trail, 1.0);
      }
    `,
  });

  divMat = new THREE.ShaderMaterial({
    uniforms: {
      uVelocity: { value: null },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uVelocity;
      uniform vec2 uResolution;
      varying vec2 vUv;
      void main() {
        vec2 t = 0.5 / uResolution;
        float l = texture2D(uVelocity, vUv - vec2(t.x, 0.0)).x;
        float r = texture2D(uVelocity, vUv + vec2(t.x, 0.0)).x;
        float b = texture2D(uVelocity, vUv - vec2(0.0, t.y)).y;
        float tp = texture2D(uVelocity, vUv + vec2(0.0, t.y)).y;
        float div = (r - l + tp - b) * 0.5;
        gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
      }
    `,
  });

  pressureMat = new THREE.ShaderMaterial({
    uniforms: {
      uPressure: { value: null },
      uDivergence: { value: null },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uPressure;
      uniform sampler2D uDivergence;
      uniform vec2 uResolution;
      varying vec2 vUv;

      void main() {
        vec2 texel = 1.0 / uResolution;
        float pL = texture2D(uPressure, vUv - vec2(texel.x, 0.0)).x;
        float pR = texture2D(uPressure, vUv + vec2(texel.x, 0.0)).x;
        float pB = texture2D(uPressure, vUv - vec2(0.0, texel.y)).x;
        float pT = texture2D(uPressure, vUv + vec2(0.0, texel.y)).x;
        float div = texture2D(uDivergence, vUv).x;
        float p = (pL + pR + pB + pT - div) * 0.25;
        gl_FragColor = vec4(p, 0.0, 0.0, 1.0);
      }
    `,
  });

  gradSubtractMat = new THREE.ShaderMaterial({
    uniforms: {
      uPressure: { value: null },
      uVelocity: { value: null },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uPressure;
      uniform sampler2D uVelocity;
      uniform vec2 uResolution;
      varying vec2 vUv;
      void main() {
        vec2 t = 0.5 / uResolution;
        float pL = texture2D(uPressure, vUv - vec2(t.x, 0.0)).x;
        float pR = texture2D(uPressure, vUv + vec2(t.x, 0.0)).x;
        float pB = texture2D(uPressure, vUv - vec2(0.0, t.y)).x;
        float pT = texture2D(uPressure, vUv + vec2(0.0, t.y)).x;
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vel -= vec2(pR - pL, pT - pB) * 0.55;
        gl_FragColor = vec4(vel, 0.0, 1.0);
      }
    `,
  });

  displayMat = new THREE.ShaderMaterial({
    uniforms: {
      uVideo: { value: null },
      uVelocity: { value: null },
      uDensity: { value: null },
      uFlow: { value: null },
      uTrail: { value: null },
      uTime: { value: 0 },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uSimResolution: { value: new THREE.Vector2(simW, simH) },
      uFingers: { value: fingerUniforms },
      uFingerCount: { value: 0 },
      uHandRadius: { value: VISUAL.handRadiusPx },
      uDistortScale: { value: VISUAL.distortScale },
      uChromaticAberration: { value: VISUAL.chromaticAberration },
      uGlassBlend: { value: VISUAL.glassBlend },
      uGlowStrength: { value: VISUAL.glowStrength },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;

      uniform sampler2D uVideo;
      uniform sampler2D uVelocity;
      uniform sampler2D uDensity;
      uniform sampler2D uFlow;
      uniform sampler2D uTrail;
      uniform float uTime;
      uniform vec2 uResolution;
      uniform vec2 uSimResolution;
      uniform vec2 uFingers[12];
      uniform float uFingerCount;
      uniform float uHandRadius;
      uniform float uDistortScale;
      uniform float uChromaticAberration;
      uniform float uGlassBlend;
      uniform float uGlowStrength;

      varying vec2 vUv;

      vec2 mirrorUv(vec2 uv) {
        return vec2(1.0 - uv.x, uv.y);
      }

      float sampleDensity(vec2 uv) {
        return texture2D(uDensity, uv).r;
      }

      vec2 gradient(vec2 uv) {
        vec2 t = 1.0 / uSimResolution;
        float l = sampleDensity(uv - vec2(t.x, 0.0));
        float r = sampleDensity(uv + vec2(t.x, 0.0));
        float b = sampleDensity(uv - vec2(0.0, t.y));
        float top = sampleDensity(uv + vec2(0.0, t.y));
        return vec2(r - l, top - b);
      }

      float handProximity(vec2 screenUv) {
        float minDist = 1e5;
        for (int i = 0; i < 12; i++) {
          if (float(i) >= uFingerCount) continue;
          vec2 delta = (screenUv - uFingers[i]) * uResolution;
          minDist = min(minDist, length(delta));
        }
        return 1.0 - smoothstep(uHandRadius * 0.35, uHandRadius * 1.05, minDist);
      }

      vec3 refractSample(vec2 uv, vec2 offset, float ca) {
        vec2 m = mirrorUv(uv);
        float r = texture2D(uVideo, mirrorUv(m + offset + vec2(ca, 0.0))).r;
        float g = texture2D(uVideo, mirrorUv(m + offset)).g;
        float b = texture2D(uVideo, mirrorUv(m + offset - vec2(ca, 0.0))).b;
        return vec3(r, g, b);
      }

      void main() {
        vec2 uv = vUv;
        vec2 simUv = uv;
        vec2 mUv = mirrorUv(uv);

        float handMask = handProximity(uv);
        float d = texture2D(uDensity, simUv).r;
        float trail = texture2D(uTrail, simUv).r;
        float localFluid = smoothstep(0.02, 0.28, max(d, trail * 0.65));
        float effectMask = handMask * localFluid;

        vec2 vel = texture2D(uVelocity, simUv).xy;
        vec2 flow = texture2D(uFlow, simUv).xy;
        vec2 grad = gradient(simUv);

        vec2 distort = (vel * 0.45 + flow * 0.35 + grad * 0.4) * uDistortScale;
        distort *= effectMask;

        float ripplePhase = trail * 12.0 - uTime * 3.0;
        distort += grad * sin(ripplePhase) * trail * 0.004 * effectMask;

        float ca = uChromaticAberration * (1.0 + length(distort) * 50.0);
        vec3 glassCol = refractSample(uv, distort, ca);
        vec3 rawVideo = texture2D(uVideo, mUv).rgb;
        float blend = mix(0.08, uGlassBlend, effectMask);
        vec3 col = mix(rawVideo, glassCol, blend);

        float edge = smoothstep(0.05, 0.22, localFluid) * (1.0 - smoothstep(0.22, 0.5, localFluid));
        col += vec3(0.0, 0.75, 0.9) * edge * effectMask * uGlowStrength;

        float vignette = 1.0 - dot((uv - 0.5) * 1.1, (uv - 0.5) * 1.1);
        col *= 0.88 + vignette * 0.12;

        gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
      }
    `,
  });

  bloomMat = new THREE.ShaderMaterial({
    uniforms: {
      uTexture: { value: null },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uThreshold: { value: 0.55 },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uTexture;
      uniform vec2 uResolution;
      uniform float uThreshold;
      varying vec2 vUv;

      void main() {
        vec3 col = texture2D(uTexture, vUv).rgb;
        float lum = max(max(col.r, col.g), col.b);
        vec3 bloom = col * smoothstep(uThreshold, uThreshold + 0.35, lum);
        gl_FragColor = vec4(bloom, 1.0);
      }
    `,
  });

  blurMat = new THREE.ShaderMaterial({
    uniforms: {
      uTexture: { value: null },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uDirection: { value: new THREE.Vector2(1, 0) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uTexture;
      uniform vec2 uResolution;
      uniform vec2 uDirection;
      varying vec2 vUv;

      void main() {
        vec2 texel = uDirection / uResolution;
        vec3 sum = vec3(0.0);
        sum += texture2D(uTexture, vUv - 4.0 * texel).rgb * 0.05;
        sum += texture2D(uTexture, vUv - 3.0 * texel).rgb * 0.09;
        sum += texture2D(uTexture, vUv - 2.0 * texel).rgb * 0.12;
        sum += texture2D(uTexture, vUv - texel).rgb * 0.15;
        sum += texture2D(uTexture, vUv).rgb * 0.18;
        sum += texture2D(uTexture, vUv + texel).rgb * 0.15;
        sum += texture2D(uTexture, vUv + 2.0 * texel).rgb * 0.12;
        sum += texture2D(uTexture, vUv + 3.0 * texel).rgb * 0.09;
        sum += texture2D(uTexture, vUv + 4.0 * texel).rgb * 0.05;
        gl_FragColor = vec4(sum, 1.0);
      }
    `,
  });

  motionBlurMat = new THREE.ShaderMaterial({
    uniforms: {
      uCurrent: { value: null },
      uPrev: { value: null },
      uVelocity: { value: null },
      uBlend: { value: 0.22 },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uSimResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uCurrent;
      uniform sampler2D uPrev;
      uniform sampler2D uVelocity;
      uniform float uBlend;
      uniform vec2 uResolution;
      uniform vec2 uSimResolution;
      varying vec2 vUv;

      void main() {
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vec2 off = vel / uSimResolution * 3.5;
        vec3 prev = texture2D(uPrev, vUv - off).rgb;
        vec3 cur = texture2D(uCurrent, vUv).rgb;
        vec3 col = mix(cur, prev, uBlend);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });

  compositeMat = new THREE.ShaderMaterial({
    uniforms: {
      uBase: { value: null },
      uBloom: { value: null },
      uBloomStrength: { value: 0.7 },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uBase;
      uniform sampler2D uBloom;
      uniform float uBloomStrength;
      varying vec2 vUv;
      void main() {
        vec3 base = texture2D(uBase, vUv).rgb;
        vec3 bloom = texture2D(uBloom, vUv).rgb;
        gl_FragColor = vec4(base + bloom * uBloomStrength, 1.0);
      }
    `,
  });

  copyMat = new THREE.ShaderMaterial({
    uniforms: { uTexture: { value: null } },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uTexture;
      varying vec2 vUv;
      void main() { gl_FragColor = texture2D(uTexture, vUv); }
    `,
  });
}

function initFluid() {
  const mobile = isMobileDevice();
  const base = mobile ? 192 : 320;
  const aspect = window.innerWidth / Math.max(window.innerHeight, 1);
  simH = base;
  simW = Math.round(base * Math.min(Math.max(aspect, 0.6), 1.8));

  rtType = getRenderTargetType(renderer);

  velocityPP = new PingPong(simW, simH, { type: rtType });
  densityPP = new PingPong(simW, simH, { type: rtType });
  flowPP = new PingPong(simW, simH, { type: rtType });
  trailPP = new PingPong(simW, simH, { type: rtType });
  pressurePP = new PingPong(simW, simH, { type: rtType });

  const screenType = THREE.UnsignedByteType;
  displayRT = new THREE.WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, type: screenType }
  );
  bloomRT = new THREE.WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, type: screenType }
  );
  prevFrameRT = new THREE.WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, type: screenType }
  );
  finalRT = new THREE.WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, type: screenType }
  );

  velocityPP.clear(renderer);
  densityPP.clear(renderer);
  flowPP.clear(renderer);
  trailPP.clear(renderer);
  pressurePP.clear(renderer);
}

function initThree() {
  if (!clock) clock = new THREE.Clock();
  if (!orthoCam) {
    orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x000000, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  scene = new THREE.Scene();
  camera = orthoCam;

  videoTexture = new THREE.VideoTexture(video);
  videoTexture.minFilter = THREE.LinearFilter;
  videoTexture.magFilter = THREE.LinearFilter;
  if (THREE.SRGBColorSpace) {
    videoTexture.colorSpace = THREE.SRGBColorSpace;
  }

  const geometry = new THREE.PlaneGeometry(2, 2);
  quad = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  scene.add(quad);

  initFluid();
  initShaders();
  initDivergenceRT();

  resize();
  window.removeEventListener("resize", resize);
  window.addEventListener("resize", resize);
  window.removeEventListener("orientationchange", onOrientationChange);
  window.addEventListener("orientationchange", onOrientationChange);
}

function onOrientationChange() {
  setTimeout(resize, 200);
}

function resize() {
  if (!renderer) return;

  const w = window.innerWidth;
  const h = window.innerHeight;

  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  if (displayRT) {
    displayRT.setSize(w, h);
    bloomRT.setSize(w, h);
    prevFrameRT.setSize(w, h);
    finalRT.setSize(w, h);
  }

  if (displayMat?.uniforms?.uResolution) {
    displayMat.uniforms.uResolution.value.set(w, h);
  }
  if (bloomMat?.uniforms?.uResolution) {
    bloomMat.uniforms.uResolution.value.set(w, h);
  }
  if (blurMat?.uniforms?.uResolution) {
    blurMat.uniforms.uResolution.value.set(w, h);
  }
  if (motionBlurMat?.uniforms?.uResolution) {
    motionBlurMat.uniforms.uResolution.value.set(w, h);
  }
}

let divergenceRT;

function initDivergenceRT() {
  divergenceRT = new THREE.WebGLRenderTarget(simW, simH, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    type: rtType || THREE.UnsignedByteType,
  });
}

function projectVelocity() {
  divMat.uniforms.uVelocity.value = velocityPP.read.texture;
  blit(divMat, divergenceRT);

  pressurePP.clear(renderer);

  for (let i = 0; i < 12; i++) {
    pressureMat.uniforms.uPressure.value = pressurePP.read.texture;
    pressureMat.uniforms.uDivergence.value = divergenceRT.texture;
    blit(pressureMat, pressurePP.write);
    pressurePP.swap();
  }

  gradSubtractMat.uniforms.uPressure.value = pressurePP.read.texture;
  gradSubtractMat.uniforms.uVelocity.value = velocityPP.read.texture;
  blit(gradSubtractMat, velocityPP.write);
  velocityPP.swap();
}

function updateSplatUniforms(time) {
  const velPoints = splatVelMat.uniforms.uPoints.value;
  const denPoints = splatDenMat.uniforms.uPoints.value;
  let count = 0;

  for (let i = 0; i < MAX_POINTS; i++) {
    const p = points[i];
    if (!velPoints[i]) velPoints[i] = new THREE.Vector4();
    if (!denPoints[i]) denPoints[i] = new THREE.Vector4();

    const power = p.active > 0.02
      ? Math.min(0.4 + p.strength + p.active * 0.35, 2.2)
      : 0;

    if (power > 0.02) {
      velPoints[i].set(p.x, 1.0 - p.y, p.vx, -p.vy);
      velPoints[i].w = power;
      denPoints[i].copy(velPoints[i]);
      count++;
    } else {
      velPoints[i].set(0, 0, 0, 0);
      denPoints[i].set(0, 0, 0, 0);
    }
  }

  splatVelMat.uniforms.uPointCount.value = count;
  splatDenMat.uniforms.uPointCount.value = count;
  splatVelMat.uniforms.uTime.value = time;
  splatDenMat.uniforms.uTime.value = time;
}

function stepFluid(dt, time) {
  updateSplatUniforms(time);

  splatVelMat.uniforms.uVelocity.value = velocityPP.read.texture;
  blit(splatVelMat, velocityPP.write);
  velocityPP.swap();

  splatDenMat.uniforms.uDensity.value = densityPP.read.texture;
  blit(splatDenMat, densityPP.write);
  densityPP.swap();

  projectVelocity();

  advectVelMat.uniforms.uVelocity.value = velocityPP.read.texture;
  advectVelMat.uniforms.uSource.value = velocityPP.read.texture;
  advectVelMat.uniforms.uDt.value = dt;
  advectVelMat.uniforms.uDecay.value = VELOCITY_DECAY;
  blit(advectVelMat, velocityPP.write);
  velocityPP.swap();

  advectDenMat.uniforms.uVelocity.value = velocityPP.read.texture;
  advectDenMat.uniforms.uSource.value = densityPP.read.texture;
  advectDenMat.uniforms.uDt.value = dt;
  advectDenMat.uniforms.uDecay.value = DENSITY_DECAY;
  blit(advectDenMat, densityPP.write);
  densityPP.swap();

  flowMat.uniforms.uFlow.value = flowPP.read.texture;
  flowMat.uniforms.uVelocity.value = velocityPP.read.texture;
  blit(flowMat, flowPP.write);
  flowPP.swap();

  trailMat.uniforms.uTrail.value = trailPP.read.texture;
  trailMat.uniforms.uDensity.value = densityPP.read.texture;
  blit(trailMat, trailPP.write);
  trailPP.swap();
}

function pushFingerDisplay(x, y) {
  if (fingerDisplayCount >= fingerUniforms.length) return;
  fingerUniforms[fingerDisplayCount].set(mirrorX(x), y);
  fingerDisplayCount++;
}

function updateDisplayUniforms() {
  if (!displayMat) return;
  displayMat.uniforms.uFingerCount.value = fingerDisplayCount;
  displayMat.uniforms.uHandRadius.value = VISUAL.handRadiusPx;
}

function renderFrame(dt, time) {
  if (!renderer || !displayMat || !quad) return;
  stepFluid(dt, time);
  updateDisplayUniforms();

  const w = window.innerWidth;
  const h = window.innerHeight;

  displayMat.uniforms.uVideo.value = videoTexture;
  displayMat.uniforms.uVelocity.value = velocityPP.read.texture;
  displayMat.uniforms.uDensity.value = densityPP.read.texture;
  displayMat.uniforms.uFlow.value = flowPP.read.texture;
  displayMat.uniforms.uTrail.value = trailPP.read.texture;
  displayMat.uniforms.uTime.value = time;
  displayMat.uniforms.uResolution.value.set(w, h);
  displayMat.uniforms.uSimResolution.value.set(simW, simH);

  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(null);
  quad.material = displayMat;
  renderer.render(scene, orthoCam);
  renderer.setRenderTarget(prev);
}

function injectPoint(x, y, vx, vy, strength) {
  if (pointCount >= MAX_POINTS) return;
  const p = points[pointCount];
  p.x = x;
  p.y = y;
  p.vx = vx;
  p.vy = vy;
  p.strength = strength;
  p.active = Math.min(p.active * 0.88 + strength * 0.25, 1.0);
  pointCount++;
}

function processHands(results) {
  pointCount = 0;
  fingerDisplayCount = 0;
  for (let i = 0; i < MAX_POINTS; i++) {
    points[i].active *= 0.985;
    points[i].vx *= 0.85;
    points[i].vy *= 0.85;
  }

  if (!results.multiHandLandmarks?.length) {
    setStatus("未检测到手部 — 请将手掌放入画面");
    return;
  }

  const mobile = isMobileDevice();
  const strengthMult = mobile ? 1.4 : 1.0;
  setStatus(`已识别 ${results.multiHandLandmarks.length} 只手 · 缓慢移动手指感受液态拖尾`);

  for (const landmarks of results.multiHandLandmarks) {
    const palm = landmarks[PALM_INDEX];
    const palmKey = `palm-${landmarks[0].x.toFixed(2)}`;
    const prevPalm = prevPositions.get(palmKey);
    let pvx = 0;
    let pvy = 0;
    if (prevPalm) {
      pvx = palm.x - prevPalm.x;
      pvy = palm.y - prevPalm.y;
    }
    prevPositions.set(palmKey, { x: palm.x, y: palm.y });
    const palmSpeed = Math.sqrt(pvx * pvx + pvy * pvy);
    pushFingerDisplay(palm.x, palm.y);
    injectPoint(mirrorX(palm.x), palm.y, -pvx, pvy, (0.4 + palmSpeed * 45) * strengthMult);

    for (let t = 0; t < TIP_INDICES.length; t++) {
      const idx = TIP_INDICES[t];
      const lm = landmarks[idx];
      const key = `tip-${idx}-${landmarks[0].x.toFixed(2)}`;
      const prev = prevPositions.get(key);
      let vx = 0;
      let vy = 0;
      if (prev) {
        vx = lm.x - prev.x;
        vy = lm.y - prev.y;
      }
      prevPositions.set(key, { x: lm.x, y: lm.y });
      const speed = Math.sqrt(vx * vx + vy * vy);
      const base = Math.min((0.55 + speed * 120) * strengthMult, 2.4);
      pushFingerDisplay(lm.x, lm.y);
      injectPoint(mirrorX(lm.x), lm.y, -vx, vy, base);
    }

    const wrist = landmarks[0];
    pushFingerDisplay(wrist.x, wrist.y);
    injectPoint(mirrorX(wrist.x), wrist.y, 0, 0, 0.18 * strengthMult);
    const mid = landmarks[13];
    pushFingerDisplay(mid.x, mid.y);
    injectPoint(mirrorX(mid.x), mid.y, 0, 0, 0.12 * strengthMult);
  }
}

function initMediaPipe() {
  hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  const mobile = isMobileDevice();
  hands.setOptions(mobile ? {
    maxNumHands: 1,
    modelComplexity: 0,
    minDetectionConfidence: 0.35,
    minTrackingConfidence: 0.35,
  } : {
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.55,
    minTrackingConfidence: 0.45,
  });

  hands.onResults(processHands);
}

async function trackHands() {
  if (!running || !hands || handBusy) return;
  if (video.readyState < video.HAVE_CURRENT_DATA || !video.videoWidth) return;

  const mobile = isMobileDevice();
  frameCount++;
  if (mobile && frameCount % 2 !== 0) return;

  handBusy = true;
  if (handBusyTimeoutId) clearTimeout(handBusyTimeoutId);
  handBusyTimeoutId = setTimeout(() => {
    handBusy = false;
  }, 300);

  try {
    await hands.send({ image: video });
  } catch (err) {
    console.warn("手势识别错误:", err);
  } finally {
    if (handBusyTimeoutId) {
      clearTimeout(handBusyTimeoutId);
      handBusyTimeoutId = null;
    }
    handBusy = false;
  }
}

function animate() {
  if (!running || !clock) return;
  requestAnimationFrame(animate);

  const dt = Math.min(clock.getDelta(), 0.033);
  const time = clock.getElapsedTime();

  if (video.readyState >= video.HAVE_CURRENT_DATA) {
    videoTexture.needsUpdate = true;
  }

  renderFrame(dt, time);
  trackHands();
}

async function attachCameraStream() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("当前浏览器不支持摄像头 API");
  }

  const mobile = isMobileDevice();
  const attempts = [
    {
      audio: false,
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: mobile ? 480 : 1280, max: mobile ? 640 : 1920 },
        height: { ideal: mobile ? 360 : 720, max: mobile ? 480 : 1080 },
      },
    },
    { audio: false, video: { facingMode: "user" } },
    { audio: false, video: true },
  ];

  let lastError;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      lastError = err;
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        throw err;
      }
    }
  }
  throw lastError;
}

function setupVideoElement() {
  video.muted = true;
  video.defaultMuted = true;
  video.autoplay = true;
  video.playsInline = true;
  video.setAttribute("muted", "");
  video.setAttribute("playsinline", "");
  video.setAttribute("webkit-playsinline", "");
}

function showCameraFallback() {
  overlay.classList.add("hidden");
  hud.classList.remove("hidden");
  video.classList.add("camera-fallback");
  setStatus("摄像头已开启（简化模式）。特效模块加载失败，请刷新或换 Chrome 重试。");
}

function getCameraErrorMessage(err) {
  const name = err?.name || "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "摄像头权限被拒绝，请在浏览器设置中允许摄像头后刷新。";
  }
  if (name === "NotFoundError") {
    return "未找到摄像头设备。";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "摄像头被占用，请关闭其他使用摄像头的应用。";
  }
  if (name === "SecurityError") {
    return "安全限制：请使用 HTTPS 访问（Vercel 域名即可）。";
  }
  return err?.message || name || "未知错误";
}

async function start(event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  if (starting || running) return;
  if (!checkLibraries()) return;

  starting = true;
  startBtn.disabled = true;
  startBtn.textContent = "正在启动…";
  setupVideoElement();
  setStatus("正在请求摄像头权限…");

  try {
    const stream = await attachCameraStream();
    video.srcObject = stream;
    video.muted = true;
    await video.play();
  } catch (err) {
    console.error("摄像头错误:", err);
    showFatal("摄像头访问失败: " + getCameraErrorMessage(err));
    return;
  }

  setStatus("正在初始化渲染…");

  try {
    initThree();
    initMediaPipe();
  } catch (err) {
    console.error("初始化失败:", err);
    starting = false;
    showCameraFallback();
    startBtn.textContent = "已开启摄像头";
    return;
  }

  overlay.classList.add("hidden");
  hud.classList.remove("hidden");

  running = true;
  starting = false;
  frameCount = 0;
  clock.start();
  animate();
  setStatus("系统就绪 — 移动手指，液态拖尾将持续 1–2 秒");
  startBtn.textContent = "已启动";
}

function bindStartButton() {
  if (!startBtn) return;

  let lastTap = 0;
  const handleStart = (e) => {
    const now = Date.now();
    if (now - lastTap < 600) return;
    lastTap = now;
    start(e);
  };

  startBtn.addEventListener("click", handleStart);
  startBtn.addEventListener(
    "touchend",
    (e) => {
      e.preventDefault();
      handleStart(e);
    },
    { passive: false }
  );
}

function bootstrap() {
  if (!video || !canvas || !startBtn) {
    showFatal("页面元素加载失败，请刷新页面。");
    return;
  }
  if (!checkLibraries()) return;
  showEnvWarning("");
  bindStartButton();
  setStatus("点击按钮开启摄像头");
}

document.addEventListener("visibilitychange", () => {
  if (!clock) return;
  if (document.hidden) clock.stop();
  else if (running) clock.start();
});

window.LiquidHand = { bootstrap };
