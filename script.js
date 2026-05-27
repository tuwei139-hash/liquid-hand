/* global THREE, Hands */

const TIP_INDICES = [4, 8, 12, 16, 20];
const PALM_INDEX = 9;
const MAX_POINTS = 12;

const TRAIL_DECAY = 0.994;
const DENSITY_DECAY = 0.992;
const VELOCITY_DECAY = 0.976;
const FLOW_DECAY = 0.994;

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
let running = false;
let starting = false;
let handBusy = false;
let rtType = null;
let clock = new THREE.Clock();
let simW = 256;
let simH = 256;

const points = Array.from({ length: MAX_POINTS }, () => ({
  x: 0.5, y: 0.5, vx: 0, vy: 0, strength: 0, active: 0,
}));
const smoothed = Array.from({ length: MAX_POINTS }, () => ({
  x: 0.5, y: 0.5, vx: 0, vy: 0,
}));
let pointCount = 0;
let prevPositions = new Map();

let velocityPP, densityPP, flowPP, trailPP, pressurePP;
let splatVelMat, splatDenMat, advectVelMat, advectDenMat, flowMat, trailMat;
let pressureMat, divMat, gradSubtractMat, displayMat;
let divergenceRT;

const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const FULLSCREEN_VS = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function isMobileDevice() {
  return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
}

function isSecureForCamera() {
  if (window.isSecureContext) return true;
  const host = location.hostname;
  return host === "localhost" || host === "127.0.0.1" || host === "[::1]";
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

function setupVideoElement() {
  video.muted = true;
  video.defaultMuted = true;
  video.autoplay = true;
  video.playsInline = true;
  video.setAttribute("muted", "");
  video.setAttribute("playsinline", "");
  video.setAttribute("webkit-playsinline", "");
}

function getCameraErrorMessage(err) {
  const name = err?.name || "";
  const msg = err?.message || String(err);

  if (err?.message === "HTTPS_REQUIRED") {
    return "手机摄像头需要 HTTPS 或 localhost 访问。请使用 https:// 链接打开，勿用 file:// 或局域网 http。";
  }
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "摄像头权限被拒绝。请在浏览器设置中允许摄像头，然后刷新重试。";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "未检测到摄像头设备。";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "摄像头被占用或无法读取。请关闭其他使用摄像头的应用后重试。";
  }
  if (name === "OverconstrainedError") {
    return "摄像头参数不支持，正在尝试兼容模式…";
  }
  if (name === "SecurityError") {
    return "安全限制：请通过 HTTPS 或 localhost 打开页面。";
  }
  if (msg.includes("mediaDevices") || msg.includes("getUserMedia")) {
    return "当前浏览器不支持摄像头 API，请使用 Chrome / Edge / Safari 最新版。";
  }
  return `摄像头错误：${msg || name || "未知错误"}`;
}

function getCameraConstraints() {
  const mobile = isMobileDevice();
  return {
    audio: false,
    video: {
      facingMode: { ideal: "user" },
      width: { ideal: mobile ? 640 : 1280, max: 1280 },
      height: { ideal: mobile ? 480 : 720, max: 720 },
    },
  };
}

function requestCameraStream() {
  if (!isSecureForCamera()) {
    return Promise.reject(new Error("HTTPS_REQUIRED"));
  }

  const constraints = getCameraConstraints();

  if (navigator.mediaDevices?.getUserMedia) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  const legacy =
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia;

  if (!legacy) {
    return Promise.reject(new Error("当前浏览器不支持 getUserMedia"));
  }

  return new Promise((resolve, reject) => {
    legacy.call(navigator, constraints, resolve, reject);
  });
}

async function attachCameraStream() {
  const constraintsList = [
    getCameraConstraints(),
    { audio: false, video: { facingMode: "user" } },
    { audio: false, video: true },
  ];

  let lastError;
  for (const constraints of constraintsList) {
    try {
      const stream = navigator.mediaDevices?.getUserMedia
        ? await navigator.mediaDevices.getUserMedia(constraints)
        : await requestCameraStream();
      return stream;
    } catch (err) {
      lastError = err;
      if (err?.message === "HTTPS_REQUIRED") throw err;
      if (err?.name === "NotAllowedError") throw err;
    }
  }
  throw lastError;
}

async function playVideoStream(stream) {
  if (video.srcObject !== stream) {
    video.srcObject = stream;
  }
  video.muted = true;
  video.setAttribute("muted", "");

  try {
    await video.play();
  } catch (playErr) {
    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
      setTimeout(resolve, 500);
    });
    await video.play();
  }
}

function getRenderTargetType(rendererInstance) {
  return rendererInstance.capabilities.isWebGL2
    ? THREE.HalfFloatType
    : THREE.UnsignedByteType;
}

function checkPageEnvironment() {
  if (typeof THREE === "undefined") {
    showEnvWarning("Three.js 加载失败，请检查网络后刷新。");
    if (startBtn) startBtn.disabled = true;
    return false;
  }
  if (typeof Hands === "undefined") {
    showEnvWarning("MediaPipe 加载失败，请检查网络后刷新。");
    if (startBtn) startBtn.disabled = true;
    return false;
  }
  if (!isSecureForCamera()) {
    showEnvWarning(
      "手机摄像头需要 HTTPS 或 localhost。请使用 https:// 地址访问（Android Chrome / Edge / Safari 均需要）。"
    );
  } else {
    showEnvWarning("");
  }
  return true;
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

  clear(renderer) {
    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(this.read);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.setRenderTarget(this.write);
    renderer.clear();
    renderer.setRenderTarget(prev);
  }
}

function blit(material, target, resW, resH) {
  if (!renderer || !quad) return;
  if (material.uniforms?.uResolution) {
    material.uniforms.uResolution.value.set(resW ?? simW, resH ?? simH);
  }
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(target);
  quad.material = material;
  renderer.render(scene, orthoCam);
  renderer.setRenderTarget(prev);
}

function getScreenSize() {
  return { w: window.innerWidth, h: window.innerHeight };
}

function resize() {
  if (!renderer) return;
  const { w, h } = getScreenSize();
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  if (displayMat) {
    displayMat.uniforms.uResolution.value.set(w, h);
    displayMat.uniforms.uHandRadius.value = VISUAL.handRadiusPx;
  }
  updateSimResolutionUniforms();
}

function updateSimResolutionUniforms() {
  const mats = [
    splatVelMat, splatDenMat, advectVelMat, advectDenMat,
    flowMat, trailMat, divMat, pressureMat, gradSubtractMat,
  ];
  for (const mat of mats) {
    if (mat?.uniforms?.uResolution) {
      mat.uniforms.uResolution.value.set(simW, simH);
    }
  }
  if (displayMat?.uniforms?.uSimResolution) {
    displayMat.uniforms.uSimResolution.value.set(simW, simH);
  }
}

function mirrorX(x) {
  return 1.0 - x;
}

function initShaders() {
  splatVelMat = new THREE.ShaderMaterial({
    uniforms: {
      uVelocity: { value: null },
      uPoints: { value: Array.from({ length: MAX_POINTS }, () => new THREE.Vector4()) },
      uPointCount: { value: 0 },
      uTime: { value: 0 },
      uRadius: { value: VISUAL.splatRadius },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uVelocity;
      uniform vec4 uPoints[12];
      uniform int uPointCount;
      uniform float uTime;
      uniform float uRadius;
      varying vec2 vUv;

      float gauss(float d, float r) {
        return exp(-d * d / (r * r + 1e-5));
      }

      void main() {
        vec2 vel = texture2D(uVelocity, vUv).xy;
        for (int i = 0; i < 12; i++) {
          if (i >= uPointCount) break;
          vec4 p = uPoints[i];
          if (p.w < 0.02) continue;
          vec2 d = vUv - p.xy;
          float dist = length(d);
          float r = uRadius * (0.5 + p.w * 0.2);
          float g = gauss(dist, r);
          vec2 force = p.zw * 5.0;
          vec2 tan = vec2(-d.y, d.x) / max(dist, 1e-4);
          vel += (force + tan * 0.08 * sin(uTime * 3.0 + float(i))) * g * p.w * 0.5;
        }
        gl_FragColor = vec4(vel, 0.0, 1.0);
      }
    `,
  });

  splatDenMat = new THREE.ShaderMaterial({
    uniforms: {
      uDensity: { value: null },
      uPoints: { value: Array.from({ length: MAX_POINTS }, () => new THREE.Vector4()) },
      uPointCount: { value: 0 },
      uTime: { value: 0 },
      uRadius: { value: VISUAL.splatRadius },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uDensity;
      uniform vec4 uPoints[12];
      uniform int uPointCount;
      uniform float uTime;
      uniform float uRadius;
      varying vec2 vUv;

      float gauss(float d, float r) {
        return exp(-d * d / (r * r + 1e-5));
      }

      void main() {
        float dens = texture2D(uDensity, vUv).r;
        for (int i = 0; i < 12; i++) {
          if (i >= uPointCount) break;
          vec4 p = uPoints[i];
          if (p.w < 0.02) continue;
          vec2 d = vUv - p.xy;
          float dist = length(d);
          float r = uRadius * (0.5 + p.w * 0.18);
          float g = gauss(dist, r);
          float gW = gauss(dist, r * 1.6);
          dens += g * (0.045 + p.w * 0.03);
          dens += sin(dist * 70.0 - uTime * 7.0) * gW * 0.022 * p.w;
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
      varying vec2 vUv;
      void main() {
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vec2 uv = clamp(vUv - vel * uDt * 1.5, 0.002, 0.998);
        gl_FragColor = vec4(texture2D(uSource, uv).xy * uDecay, 0.0, 1.0);
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
      varying vec2 vUv;
      void main() {
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vec2 uv = clamp(vUv - vel * uDt * 1.3, 0.002, 0.998);
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
        vec2 f = texture2D(uFlow, vUv).xy * uDecay;
        vec2 v = texture2D(uVelocity, vUv).xy;
        float sp = length(v);
        f += (sp > 1e-4 ? normalize(v) * sp * 0.3 : vec2(0.0));
        gl_FragColor = vec4(clamp(f, -1.5, 1.5), length(f), 1.0);
      }
    `,
  });

  trailMat = new THREE.ShaderMaterial({
    uniforms: {
      uTrail: { value: null },
      uDensity: { value: null },
      uDecay: { value: TRAIL_DECAY },
      uRetain: { value: VISUAL.trailRetain },
      uResolution: { value: new THREE.Vector2(simW, simH) },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uTrail;
      uniform sampler2D uDensity;
      uniform float uDecay;
      uniform float uRetain;
      varying vec2 vUv;
      void main() {
        float prev = texture2D(uTrail, vUv).r * uDecay;
        float dens = texture2D(uDensity, vUv).r;
        float t = max(prev, dens * uRetain);
        gl_FragColor = vec4(t, t, t, 1.0);
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
        gl_FragColor = vec4((r - l + tp - b) * 0.5, 0.0, 0.0, 1.0);
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
        vec2 t = 1.0 / uResolution;
        float pL = texture2D(uPressure, vUv - vec2(t.x, 0.0)).x;
        float pR = texture2D(uPressure, vUv + vec2(t.x, 0.0)).x;
        float pB = texture2D(uPressure, vUv - vec2(0.0, t.y)).x;
        float pT = texture2D(uPressure, vUv + vec2(0.0, t.y)).x;
        float div = texture2D(uDivergence, vUv).x;
        gl_FragColor = vec4((pL + pR + pB + pT - div) * 0.25, 0.0, 0.0, 1.0);
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
        vel -= vec2(pR - pL, pT - pB) * 0.5;
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
      uFingers: { value: Array.from({ length: MAX_POINTS }, () => new THREE.Vector3()) },
      uFingerCount: { value: 0 },
      uTime: { value: 0 },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uSimResolution: { value: new THREE.Vector2(simW, simH) },
      uHandRadius: { value: VISUAL.handRadiusPx },
      uDistortScale: { value: VISUAL.distortScale },
      uCA: { value: VISUAL.chromaticAberration },
      uGlassBlend: { value: VISUAL.glassBlend },
      uGlow: { value: VISUAL.glowStrength },
    },
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
      precision highp float;

      uniform sampler2D uVideo;
      uniform sampler2D uVelocity;
      uniform sampler2D uDensity;
      uniform sampler2D uFlow;
      uniform sampler2D uTrail;
      uniform vec3 uFingers[12];
      uniform int uFingerCount;
      uniform float uTime;
      uniform vec2 uResolution;
      uniform vec2 uSimResolution;
      uniform float uHandRadius;
      uniform float uDistortScale;
      uniform float uCA;
      uniform float uGlassBlend;
      uniform float uGlow;

      varying vec2 vUv;

      vec2 videoUv(vec2 uv) {
        return vec2(1.0 - uv.x, uv.y);
      }

      float handProximity(vec2 uv) {
        vec2 px = uv * uResolution;
        float zone = 0.0;
        for (int i = 0; i < 12; i++) {
          if (i >= uFingerCount) break;
          vec3 f = uFingers[i];
          if (f.z < 0.02) continue;
          vec2 fp = f.xy * uResolution;
          float d = length(px - fp);
          float inner = uHandRadius * 0.45;
          float outer = uHandRadius;
          zone = max(zone, 1.0 - smoothstep(inner, outer, d));
        }
        return zone;
      }

      float sampleFluid(vec2 uv) {
        float d = texture2D(uDensity, uv).r;
        float t = texture2D(uTrail, uv).r;
        return max(d, t * 0.85);
      }

      vec2 fluidGradient(vec2 uv) {
        vec2 t = 1.0 / uSimResolution;
        float l = sampleFluid(uv - vec2(t.x, 0.0));
        float r = sampleFluid(uv + vec2(t.x, 0.0));
        float b = sampleFluid(uv - vec2(0.0, t.y));
        float tp = sampleFluid(uv + vec2(0.0, t.y));
        return vec2(r - l, tp - b);
      }

      float metaballEdge(float f) {
        float blob = smoothstep(0.02, 0.14, f);
        float edge = smoothstep(0.06, 0.18, f) - smoothstep(0.18, 0.35, f);
        return blob * 0.4 + edge * 0.6;
      }

      vec3 sampleVideo(vec2 uv, vec2 off, float ca) {
        vec2 base = videoUv(uv);
        float r = texture2D(uVideo, videoUv(uv + off + vec2(ca, 0.0))).r;
        float g = texture2D(uVideo, videoUv(uv + off)).g;
        float b = texture2D(uVideo, videoUv(uv + off - vec2(ca, 0.0))).b;
        return vec3(r, g, b);
      }

      void main() {
        vec2 uv = vUv;
        vec3 camera = sampleVideo(uv, vec2(0.0), 0.0);

        float handZone = handProximity(uv);
        float fluid = sampleFluid(uv);
        float meta = metaballEdge(fluid);

        float effectMask = handZone * smoothstep(0.004, 0.12, fluid + handZone * 0.04);
        effectMask = clamp(effectMask, 0.0, 1.0);

        if (effectMask < 0.001) {
          gl_FragColor = vec4(camera, 1.0);
          return;
        }

        vec2 vel = texture2D(uVelocity, uv).xy;
        vec2 flow = texture2D(uFlow, uv).xy;
        vec2 grad = fluidGradient(uv);
        float trail = texture2D(uTrail, uv).r;

        vec2 distort = (vel * 0.6 + flow * 0.35 + grad * 0.5) * uDistortScale;
        distort += grad * meta * uDistortScale * 0.8;
        float ripple = sin(trail * 22.0 - uTime * 4.0) * trail * 0.004;
        distort += grad * ripple;
        float dLen = length(distort);
        distort *= min(1.0, uDistortScale * 1.2 / max(dLen, 1e-4));

        float ca = uCA * (1.0 + dLen * 40.0);
        vec3 glass = sampleVideo(uv, distort * effectMask, ca * effectMask);

        float fresnel = pow(1.0 - meta, 2.0) * meta;
        vec3 tint = vec3(0.55, 0.78, 0.92) * fresnel * uGlow;
        float rippleRing = sin(length(grad) * 40.0 - uTime * 5.0) * 0.5 + 0.5;
        tint += vec3(0.4, 0.65, 0.85) * rippleRing * trail * uGlow * 0.5;

        vec3 col = mix(camera, glass + tint, effectMask * uGlassBlend);

        gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
      }
    `,
  });
}

function initFluid() {
  const isMobile = isMobileDevice();
  const base = isMobile ? 224 : 384;
  const aspect = window.innerWidth / Math.max(window.innerHeight, 1);
  simH = base;
  simW = Math.round(base * Math.min(Math.max(aspect, 0.65), 1.75));

  rtType = getRenderTargetType(renderer);

  velocityPP = new PingPong(simW, simH, { type: rtType });
  densityPP = new PingPong(simW, simH, { type: rtType });
  flowPP = new PingPong(simW, simH, { type: rtType });
  trailPP = new PingPong(simW, simH, { type: rtType });
  pressurePP = new PingPong(simW, simH, { type: rtType });

  velocityPP.clear(renderer);
  densityPP.clear(renderer);
  flowPP.clear(renderer);
  trailPP.clear(renderer);
  pressurePP.clear(renderer);
}

function initDivergenceRT() {
  divergenceRT = new THREE.WebGLRenderTarget(simW, simH, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    type: rtType,
  });
}

function initThree() {
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x000000, 1);

  scene = new THREE.Scene();
  camera = orthoCam;

  videoTexture = new THREE.VideoTexture(video);
  videoTexture.minFilter = THREE.LinearFilter;
  videoTexture.magFilter = THREE.LinearFilter;
  if (THREE.SRGBColorSpace) {
    videoTexture.colorSpace = THREE.SRGBColorSpace;
  }

  quad = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial()
  );
  scene.add(quad);

  initFluid();
  initShaders();
  initDivergenceRT();
  updateSimResolutionUniforms();
  resize();

  window.addEventListener("resize", resize);
  window.addEventListener("orientationchange", () => setTimeout(resize, 200));
}

function projectVelocity() {
  divMat.uniforms.uVelocity.value = velocityPP.read.texture;
  blit(divMat, divergenceRT);
  pressurePP.clear(renderer);
  for (let i = 0; i < 10; i++) {
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
  const velPts = splatVelMat.uniforms.uPoints.value;
  const denPts = splatDenMat.uniforms.uPoints.value;
  let count = 0;

  for (let i = 0; i < MAX_POINTS; i++) {
    const p = points[i];
    if (!velPts[i]) velPts[i] = new THREE.Vector4();

    const power = p.active > 0.03
      ? Math.min(0.25 + p.strength * 0.2 + p.active * 0.35, 1.0)
      : 0;

    if (power > 0.03) {
      const simY = 1.0 - p.y;
      velPts[i].set(p.x, simY, p.vx, -p.vy);
      velPts[i].w = power;
      denPts[i].copy(velPts[i]);
      count++;
    } else {
      velPts[i].set(0, 0, 0, 0);
      denPts[i].set(0, 0, 0, 0);
    }
  }

  splatVelMat.uniforms.uPointCount.value = count;
  splatDenMat.uniforms.uPointCount.value = count;
  splatVelMat.uniforms.uTime.value = time;
  splatDenMat.uniforms.uTime.value = time;
}

function updateDisplayUniforms() {
  const fingers = displayMat.uniforms.uFingers.value;
  let count = 0;

  for (let i = 0; i < MAX_POINTS; i++) {
    if (!fingers[i]) fingers[i] = new THREE.Vector3();
    const p = points[i];
    if (p.active > 0.03) {
      fingers[i].set(p.x, 1.0 - p.y, p.active);
      count++;
    } else {
      fingers[i].set(0, 0, 0);
    }
  }
  displayMat.uniforms.uFingerCount.value = count;
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
  blit(advectVelMat, velocityPP.write);
  velocityPP.swap();

  advectDenMat.uniforms.uVelocity.value = velocityPP.read.texture;
  advectDenMat.uniforms.uSource.value = densityPP.read.texture;
  advectDenMat.uniforms.uDt.value = dt;
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

function renderFrame(dt, time) {
  if (!renderer || !displayMat) return;

  stepFluid(dt, time);
  updateDisplayUniforms();

  const { w, h } = getScreenSize();
  displayMat.uniforms.uVideo.value = videoTexture;
  displayMat.uniforms.uVelocity.value = velocityPP.read.texture;
  displayMat.uniforms.uDensity.value = densityPP.read.texture;
  displayMat.uniforms.uFlow.value = flowPP.read.texture;
  displayMat.uniforms.uTrail.value = trailPP.read.texture;
  displayMat.uniforms.uTime.value = time;

  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(null);
  quad.material = displayMat;
  renderer.render(scene, camera);
  renderer.setRenderTarget(prev);
}

function injectPoint(mx, y, vx, vy, strength) {
  if (pointCount >= MAX_POINTS) return;

  const s = smoothed[pointCount];
  const t = VISUAL.smoothFactor;
  s.x += (mx - s.x) * t;
  s.y += (y - s.y) * t;
  s.vx += (vx - s.vx) * t;
  s.vy += (vy - s.vy) * t;

  const p = points[pointCount];
  p.x = s.x;
  p.y = s.y;
  p.vx = s.vx;
  p.vy = s.vy;
  p.strength = strength;
  p.active = Math.min(p.active * 0.94 + strength * 0.15 + 0.08, 1.0);
  pointCount++;
}

function processHands(results) {
  pointCount = 0;
  for (let i = 0; i < MAX_POINTS; i++) {
    points[i].active *= 0.99;
    points[i].vx *= 0.88;
    points[i].vy *= 0.88;
  }

  if (!results.multiHandLandmarks?.length) {
    setStatus("未检测到手部 — 请将手掌放入画面");
    return;
  }

  setStatus(`已识别 ${results.multiHandLandmarks.length} 只手 · 移动手指划过液态玻璃`);

  for (const landmarks of results.multiHandLandmarks) {
    const palm = landmarks[PALM_INDEX];
    const palmKey = `palm-${landmarks[0].x.toFixed(3)}`;
    const pp = prevPositions.get(palmKey);
    let pvx = 0;
    let pvy = 0;
    if (pp) {
      pvx = palm.x - pp.x;
      pvy = palm.y - pp.y;
    }
    prevPositions.set(palmKey, { x: palm.x, y: palm.y });
    const palmSpeed = Math.hypot(pvx, pvy);
    injectPoint(mirrorX(palm.x), palm.y, -pvx, pvy, 0.2 + palmSpeed * 25);

    for (const idx of TIP_INDICES) {
      if (pointCount >= MAX_POINTS) break;
      const lm = landmarks[idx];
      const key = `t${idx}-${landmarks[0].x.toFixed(3)}`;
      const prev = prevPositions.get(key);
      let vx = 0;
      let vy = 0;
      if (prev) {
        vx = lm.x - prev.x;
        vy = lm.y - prev.y;
      }
      prevPositions.set(key, { x: lm.x, y: lm.y });
      const speed = Math.hypot(vx, vy);
      injectPoint(mirrorX(lm.x), lm.y, -vx, vy, 0.18 + speed * 35);
    }
  }
}

function initMediaPipe() {
  hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: isMobileDevice() ? 0 : 1,
    minDetectionConfidence: 0.55,
    minTrackingConfidence: 0.45,
  });

  hands.onResults(processHands);
}

async function trackHands() {
  if (!running || !hands || handBusy) return;
  if (video.readyState < video.HAVE_CURRENT_DATA || !video.videoWidth) return;

  handBusy = true;
  try {
    await hands.send({ image: video });
  } catch (err) {
    console.warn("手势识别帧错误:", err);
  } finally {
    handBusy = false;
  }
}

function animate() {
  if (!running) return;
  requestAnimationFrame(animate);
  if (!renderer || !videoTexture) return;

  const dt = Math.min(clock.getDelta(), 0.033);
  const time = clock.getElapsedTime();

  if (video.readyState >= video.HAVE_CURRENT_DATA) {
    videoTexture.needsUpdate = true;
  }

  renderFrame(dt, time);
  trackHands();
}

async function start(event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  if (starting || running) return;

  if (!checkPageEnvironment()) {
    setStatus("环境检查未通过，请查看提示");
    return;
  }

  if (!isSecureForCamera()) {
    const msg = getCameraErrorMessage(new Error("HTTPS_REQUIRED"));
    showEnvWarning(msg);
    setStatus(msg);
    return;
  }

  starting = true;
  startBtn.disabled = true;
  startBtn.textContent = "正在启动…";
  setStatus("正在请求摄像头权限…");

  try {
    const stream = await attachCameraStream();
    await playVideoStream(stream);
  } catch (err) {
    console.error("摄像头错误:", err);
    const msg = getCameraErrorMessage(err);
    showEnvWarning(msg);
    setStatus(msg);
    startBtn.disabled = false;
    startBtn.textContent = "开启摄像头体验";
    starting = false;
    return;
  }

  try {
    setStatus("正在初始化渲染…");
    initThree();
    initMediaPipe();
  } catch (err) {
    console.error("初始化错误:", err);
    const msg = `初始化失败：${err.message || err}`;
    showEnvWarning(msg);
    setStatus(msg);
    if (video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    startBtn.disabled = false;
    startBtn.textContent = "开启摄像头体验";
    starting = false;
    return;
  }

  overlay.classList.add("hidden");
  hud.classList.remove("hidden");
  running = true;
  starting = false;
  clock.start();
  animate();
  setStatus("就绪 — 手划哪里，哪里出现液态拖尾");
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

function initPage() {
  setupVideoElement();
  checkPageEnvironment();
  bindStartButton();
  setStatus("点击按钮开启摄像头");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initPage);
} else {
  initPage();
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) clock.stop();
  else if (running) clock.start();
});
