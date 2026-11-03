// Rescue Listener: a drone over the January 2010 LiDAR of Port-au-Prince.
// Scene frame: metres, origin at the crop centre (see data/meta.json), x east, y up, z south.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const BG = new THREE.Color('#0A0E13');
const GREEN = new THREE.Color('#00D46A');             // the one accent
const SLATE = new THREE.Color('#2A3340');             // roof tops
const AMBER = new THREE.Color('#F5A524');             // a flag the scene cannot vouch for: never green without the truth
const GREY = new THREE.Color('#71717A');              // debris verdict: the rings go quiet
const HEAP = new THREE.Vector2(33, 92);  // the opening beat: an open yard by a collapsed roof south of the palace, 50+ m from every pile, nothing tall near
const FLIGHT_S = 42;                      // transit length before the hover
const HOVER_AGL = 14;
const DRONE_SPAN = 1.0;                   // metres across the rotors: a payload-class drone; the camera comes close instead
const params = new URLSearchParams(location.search);
const CAPTURE = params.has('capture');   // headless stills: software GL draws ~1 frame/s
if (CAPTURE) document.documentElement.dataset.capture = '1';   // CSS transitions stall at ~1 frame/s: stills show the end state

const [meta, groundBuf, intBuf, aerial, sky, droneGltf] = await Promise.all([
  fetch('data/meta.json').then(r => r.json()),
  fetch('data/ground.bin').then(r => r.arrayBuffer()),
  fetch('data/ground_i.bin').then(r => r.arrayBuffer()),
  new THREE.TextureLoader().loadAsync('data/aerial.jpg'),   // 21 Jan 2010, north up, exactly the crop (data/aerial.json)
  new RGBELoader().loadAsync('vendor/hdri/kloofendal_48d_partly_cloudy_puresky_2k.hdr'),   // Poly Haven, CC0
  new GLTFLoader().loadAsync('vendor/models/quadrocopter2.glb'),   // yanix, CC BY 4.0
]);
aerial.colorSpace = THREE.SRGBColorSpace;
const N = meta.size_px, CELL = meta.cell_m, HALF = N * CELL / 2;
const LIFT = meta.ground_med - meta.min_h;          // grid heights are over min_h; scene y=0 is the median
const Hraw = new Uint16Array(groundBuf), Iraw = new Uint8Array(intBuf);
const H = new Float32Array(N * N);
for (let i = 0; i < H.length; i++) H[i] = Hraw[i] / 100 - LIFT;

function heightAt(x, z) {                            // bilinear on the grid; row 0 = north edge
  const c = THREE.MathUtils.clamp((x + HALF) / CELL - .5, 0, N - 1.001);
  const r = THREE.MathUtils.clamp((z + HALF) / CELL - .5, 0, N - 1.001);
  const c0 = c | 0, r0 = r | 0, fc = c - c0, fr = r - r0, i = r0 * N + c0;
  return (H[i] * (1 - fc) + H[i + 1] * fc) * (1 - fr) + (H[i + N] * (1 - fc) + H[i + N + 1] * fc) * fr;
}

// ---------- renderer, scene, camera ----------
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = .95;
const scene = new THREE.Scene();
sky.mapping = THREE.EquirectangularReflectionMapping;
scene.background = new THREE.Color(.35, .376, .46);   // the horizon is the fog itself: no dome edge, no arc (the HDRI still lights)
scene.environment = sky; scene.environmentIntensity = .3;   // the photo already holds that day's light; the sky only fills
scene.fog = new THREE.FogExp2(new THREE.Color(.35, .376, .46), .0022);   // the HDRI's horizon, dimmed to sit under it
aerial.anisotropy = renderer.capabilities.getMaxAnisotropy();
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const camera = new THREE.PerspectiveCamera(38, 1, .5, 5000);

// the HDRI's sun sits at azimuth 124 deg, 48 deg up (measured on the file); the aerial's shadows fall the same way
const SUN = new THREE.Vector3(.554, .743, .375).normalize();
const sun = new THREE.DirectionalLight('#fff0dc', 1.5); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -.0004; sun.shadow.radius = 3;
scene.add(sun, sun.target);
function fitShadow(look, dist) {            // the shadow box follows the view: tight on close-ups, wide on the end shot
  const r = THREE.MathUtils.clamp(dist * .9, 14, 170), c = sun.shadow.camera;
  sun.target.position.copy(look); sun.position.copy(look).addScaledVector(SUN, 300);
  c.left = c.bottom = -r; c.right = c.top = r; c.near = 100; c.far = 500; c.updateProjectionMatrix();
  sun.shadow.normalBias = r * .004;
}


// ---------- ground: a mesh from the LiDAR DSM, shaded by height, intensity and cavity ----------
const uni = {
  uTime: { value: 0 }, uHeap: { value: HEAP }, uDrone: { value: new THREE.Vector3() },
  uListen: { value: 0 }, uRingCol: { value: new THREE.Color('#F4F4F5') }, uMark: { value: 0 }, uMarkT: { value: -1 },
  // three listening nodes (x halo on, y pulse amplitude, z ripple once the target rhythm is heard there)
  uNodes: { value: [0, 1, 2].map(() => new THREE.Vector2()) }, uNodeA: { value: [0, 1, 2].map(() => new THREE.Vector3()) },
  uNodeCol: { value: [0, 1, 2].map(() => new THREE.Color('#F4F4F5')) }, uRipple: { value: HEAP }, uRippleT: { value: -1 },
  uAerial: { value: aerial }, uSun: { value: SUN }, uBg: { value: BG }, uGreen: { value: GREEN }, uSlate: { value: SLATE },
  uHalf: { value: HALF }, uLift: { value: LIFT }, uRange: { value: meta.max_h - meta.min_h },
};
// The mesh is a 1 m average of the 0.5 m grid: it occludes and takes light; the points carry the detail.
const M = N / 2;
const groundGeo = new THREE.PlaneGeometry((M - 1) * 2 * CELL, (M - 1) * 2 * CELL, M - 1, M - 1);
groundGeo.rotateX(-Math.PI / 2);
{
  const pos = groundGeo.attributes.position;
  const Hm = new Float32Array(M * M), inten = new Float32Array(M * M);
  for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) {
    const i = 2 * r * N + 2 * c;
    Hm[r * M + c] = (H[i] + H[i + 1] + H[i + N] + H[i + N + 1]) / 4;
    inten[r * M + c] = (Iraw[i] + Iraw[i + 1] + Iraw[i + N] + Iraw[i + N + 1]) / 1020;
  }
  // single-vertex spikes (poles, tree tops) read as black cones from the high camera: cap each at its tallest neighbour
  for (let pass = 0; pass < 2; pass++) for (let r = 1; r < M - 1; r++) for (let c = 1; c < M - 1; c++) {
    const i = r * M + c, nb = Math.max(Hm[i - 1], Hm[i + 1], Hm[i - M], Hm[i + M], Hm[i - M - 1], Hm[i - M + 1], Hm[i + M - 1], Hm[i + M + 1]);
    if (Hm[i] > nb + 1.5) Hm[i] = nb;
  }
  {   // open with a 5 m square: anything narrower than that standing 3 m proud (poles, crowns, spikes) goes
    const lo = new Float32Array(M * M), op = new Float32Array(M * M), R2 = 2;
    for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) { let m = Infinity;
      for (let dr = -R2; dr <= R2; dr++) for (let dc = -R2; dc <= R2; dc++) m = Math.min(m, Hm[Math.min(Math.max(r + dr, 0), M - 1) * M + Math.min(Math.max(c + dc, 0), M - 1)]);
      lo[r * M + c] = m; }
    for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) { let m = -Infinity;
      for (let dr = -R2; dr <= R2; dr++) for (let dc = -R2; dc <= R2; dc++) m = Math.max(m, lo[Math.min(Math.max(r + dr, 0), M - 1) * M + Math.min(Math.max(c + dc, 0), M - 1)]);
      op[r * M + c] = m; }
    for (let i = 0; i < M * M; i++) if (Hm[i] - op[i] > 3) Hm[i] = op[i] + 1;
  }
  for (let i = 0; i < M * M; i++) pos.setY(i, Hm[i]);
  // cavity = height minus a 4 m box blur: dark in crevices, lit on ridges; reads rubble at a glance
  const blur = boxBlur(Hm, M, 4), cav = new Float32Array(M * M);
  for (let i = 0; i < M * M; i++) cav[i] = THREE.MathUtils.clamp((Hm[i] - blur[i]) / 3, -1, 1);
  groundGeo.setAttribute('aCav', new THREE.BufferAttribute(cav, 1));
  groundGeo.setAttribute('aInt', new THREE.BufferAttribute(inten, 1));
  groundGeo.computeVertexNormals();
}
function boxBlur(src, n, r) {
  const tmp = new Float32Array(n * n), out = new Float32Array(n * n);
  for (let pass = 0; pass < 2; pass++) {
    const a = pass ? tmp : src, b = pass ? out : tmp;
    for (let y = 0; y < n; y++) {
      let s = 0;
      for (let x = -r; x <= r; x++) s += a[pass ? Math.min(Math.max(x, 0), n - 1) * n + y : y * n + Math.min(Math.max(x, 0), n - 1)];
      for (let x = 0; x < n; x++) {
        b[pass ? x * n + y : y * n + x] = s / (2 * r + 1);
        const add = Math.min(x + r + 1, n - 1), sub = Math.max(x - r, 0);
        s += a[pass ? add * n + y : y * n + add] - a[pass ? sub * n + y : y * n + sub];
      }
    }
  }
  return out;
}

const RING_GLSL = `
  // the listening ring around the heap, the mark's shock ring, and the drone's light pool
  float ripple(float d) {                     // two thin rings every 1.8 s out to 24 m, fading in over 0.6 s
    float a = fract(uRippleT / 1.8), b = fract(uRippleT / 1.8 + .5);
    return (exp(-pow((d - a * 24.) / .4, 2.)) * (1. - a) + exp(-pow((d - b * 24.) / .4, 2.)) * (1. - b) * .5) * smoothstep(0., .6, uRippleT);
  }
  vec3 overlays(vec3 wp, vec3 col) {
    float d = distance(wp.xz, uHeap);
    col += uRingCol * exp(-pow((d - 7.) / .15, 2.)) * .9 * uListen;   // one thin ring where the cone meets the ground
    float m = uMarkT < 0. ? 0. : uMarkT;
    float shock = exp(-pow((d - m * 14.) / .6, 2.)) * exp(-m * 1.3);
    float pad = (exp(-pow((d - 2.2) / .14, 2.)) + smoothstep(1.4, 0., d) * .35) * (.75 + .25 * sin(uTime * 3.));
    col += uGreen * (shock * 2.5 + pad * 1.4) * uMark;
    // the listening nodes: one quiet ring every 2.4 s, in step on every pile
    float wn = fract(uTime / 2.4);
    for (int i = 0; i < 3; i++) {
      float dn = distance(wp.xz, uNodes[i]);
      col += uNodeCol[i] * (exp(-pow((dn - wn * 9.) / .3, 2.)) * (1. - wn) * 1.2 * uNodeA[i].y + exp(-pow((dn - 1.6) / .12, 2.)) * .5 * uNodeA[i].x);
      if (uRippleT >= 0.) col += uGreen * ripple(dn) * uNodeA[i].z;
    }
    // the mark's ripple: rings pushing out from under the rubble
    if (uRippleT >= 0.) col += uGreen * ripple(distance(wp.xz, uRipple)) * uMark;
    return col;
  }`;

// The ground is a lit standard material wearing the 2010 aerial, so it takes the HDRI, the sun's shadows and
// the AO pass; the listening rings, ripples and node halos are added as emission.
{ const k = (M - 1) * CELL / HALF;   // the mesh spans 299 m, the photo the full 300 m crop
  aerial.repeat.set(k, k); aerial.offset.setScalar((1 - k) / 2); }
const groundMat = new THREE.MeshStandardMaterial({ map: aerial, roughness: .95, metalness: 0, flatShading: true,
  polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 2 });
groundMat.onBeforeCompile = sh => {
  Object.assign(sh.uniforms, uni);
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', '#include <common>\nattribute float aCav; varying float vCav; varying vec3 vW;')
    .replace('#include <project_vertex>', '#include <project_vertex>\nvCav = aCav; vW = (modelMatrix * vec4(transformed, 1.)).xyz;');
  sh.fragmentShader = sh.fragmentShader
    .replace('#include <common>', `#include <common>
      uniform float uTime, uListen, uMark, uMarkT, uHalf, uRippleT; uniform vec2 uHeap, uRipple; uniform vec2 uNodes[3];
      uniform vec3 uNodeA[3], uNodeCol[3], uGreen, uRingCol; varying float vCav; varying vec3 vW;
      ${RING_GLSL}`)
    .replace('#include <map_fragment>', `#include <map_fragment>
      if (length(vW.xz) > uHalf - 2.) discard;               // the square crop, clipped to a clean circle
      {  // the camp's red tarps read as render smears, or as survivor hits: all the way to grey
        float red = smoothstep(.06, .2, diffuseColor.r - max(diffuseColor.g, diffuseColor.b));   // fully grey: a red dot reads as a thermal hit
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(dot(diffuseColor.rgb, vec3(.3, .59, .11))), red);
      }
      {  // walls smear the top-down photo: give them flat concrete; cavity darkens the crevices
        float top = step(.6, normalize(cross(dFdx(vW), dFdy(vW))).y);
        diffuseColor.rgb = mix(vec3(.13, .125, .118), diffuseColor.rgb, top)   // walls: flat dark concrete, not the photo stretched down them
          * clamp(.85 + vCav * .4, .5, 1.15);
      }`)
    .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += overlays(vW, vec3(0.)) * smoothstep(.4, .8, normalize(cross(dFdx(vW), dFdy(vW))).y);   // rings lie on the ground, never smeared up a wall')
    .replace('#include <fog_fragment>', `#include <fog_fragment>
      gl_FragColor.rgb = mix(fogColor, gl_FragColor.rgb, smoothstep(uHalf - 2., uHalf - 40., length(vW.xz)));   // a wide rim into the haze`);
};
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.castShadow = ground.receiveShadow = true;
scene.add(ground);
{ // the haze skirt: from the crop's edge to past the far plane, so the disc never ends against the sky
  let y = Infinity;   // under the lowest rim point: where the rim dips below the skirt, rays slip under it to the sky
  for (let i = 0; i < 256; i++) y = Math.min(y, heightAt(Math.cos(i / 256 * 6.283) * (HALF - 2.5), Math.sin(i / 256 * 6.283) * (HALF - 2.5)));
  const skirt = new THREE.Mesh(new THREE.RingGeometry(HALF - 2, 4000, 96, 1), new THREE.MeshBasicMaterial({ color: scene.fog.color }));
  skirt.rotation.x = -Math.PI / 2; skirt.position.y = y - 1; scene.add(skirt);
}

// ---------- the drone: a low-poly quad, light on the dark city so it reads at any distance ----------
const drone = new THREE.Group();
const props = [];
drone.rotation.order = "YXZ";
let SKID;                                                             // centre to the bottom of the landing guard
{
  const m = droneGltf.scene;                // "Quadrocopter 2" by yanix, CC BY 4.0 (README)
  const box = new THREE.Box3().setFromObject(m), size = box.getSize(new THREE.Vector3()), mid = box.getCenter(new THREE.Vector3());
  const k = DRONE_SPAN / Math.max(size.x, size.z);
  m.scale.setScalar(k); m.position.copy(mid).multiplyScalar(-k);   // centred on the drone's origin
  drone.add(m); drone.updateMatrixWorld(true);
  SKID = size.y * k / 2;
  const up = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
  for (let i = 1; i <= 4; i++) {
    const node = m.getObjectByName(`quadrocopter_propeller_${i}_object`);
    const axis = up.clone().applyQuaternion(node.getWorldQuaternion(q).invert()).normalize();   // world up, in the prop's own frame
    const disc = new THREE.Mesh(new THREE.CircleGeometry(DRONE_SPAN * .19, 24), new THREE.MeshBasicMaterial({ color: '#F4F4F5', transparent: true, opacity: .1, depthWrite: false, side: THREE.DoubleSide }));
    disc.rotation.x = -Math.PI / 2; disc.position.copy(drone.worldToLocal(node.getWorldPosition(new THREE.Vector3())));
    drone.add(disc); props.push({ node, axis, disc });
  }
  const led = new THREE.Mesh(new THREE.OctahedronGeometry(.02, 0), new THREE.MeshBasicMaterial({ color: GREEN.clone().multiplyScalar(3) }));
  led.position.set(0, SKID + .01, 0); drone.add(led);
}
// the drone's acoustic array ring, lit while listening
const arrayRing = new THREE.Mesh(new THREE.TorusGeometry(DRONE_SPAN * .22, .006, 6, 64), new THREE.MeshBasicMaterial({ color: GREEN.clone().multiplyScalar(2.2), transparent: true, opacity: 0 }));
arrayRing.rotation.x = Math.PI / 2; arrayRing.position.y = -SKID - .03; drone.add(arrayRing);
// the listening cone, drawn as its two outline edges (set each frame in coneEdges)
const cone = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3)),
  new THREE.LineBasicMaterial({ color: GREEN.clone(), transparent: true, opacity: 0, depthWrite: false }));
cone.frustumCulled = false;
const CONE_R = 7;   // metres at the ground: the same 7 m as the ground ring
// The generators of a cone (apex a, base circle of radius CONE_R at height gy) that graze the line of sight from e:
// where e lies in the cone's tangent plane along the generator, i.e. (e - a) . ((p - a) x t) changes sign.
function coneEdges(a, gy, e) {
  const pos = cone.geometry.attributes.position;
  let k = 0, prev = 0;
  for (let i = 0; i <= 192 && k < 2; i++) {
    const th = i / 192 * Math.PI * 2, c = Math.cos(th), sn = Math.sin(th);
    const px = CONE_R * c, py = gy - a.y, pz = CONE_R * sn;   // p - a; the tangent t is (-sn, 0, c)
    const nx = py * c, ny = pz * -sn - px * c, nz = py * sn;   // (p - a) x t
    const f = (e.x - a.x) * nx + (e.y - a.y) * ny + (e.z - a.z) * nz;
    if (i && Math.sign(f) !== Math.sign(prev)) { pos.setXYZ(2 * k, a.x, a.y, a.z); pos.setXYZ(2 * k + 1, a.x + px, gy, a.z + pz); k++; }
    prev = f;
  }
  pos.needsUpdate = true;
  return k === 2;   // the camera inside the cone has no outline
}
scene.add(drone, cone);
drone.traverse(o => { if (o.isMesh) o.castShadow = true; });

// ---------- listening nodes: the drone sets one on each pile; a mast beacon blinks the same on every node ----------
// Each pile sits on a rough DSM bump 3-5 m above a 20 m opening (a collapse, not a road or a flat roof), searched near
// the piles first picked by eye on the Jan 2010 aerial.
const PILES = [new THREE.Vector2(-47, 15), new THREE.Vector2(10, -28), new THREE.Vector2(48, -5)];
const INK = new THREE.Color('#F4F4F5');
// a microphone glyph in ink on every node callout: the node is a microphone, nothing more
const MIC = '<svg class="mic" viewBox="0 0 12 16" aria-hidden="true"><rect x="3.5" y="1" width="5" height="9" rx="2.5"/><path d="M1.5 7.5a4.5 4.5 0 0 0 9 0M6 12v3M3.5 15h5"/></svg>';
const nodes = [];
for (let i = 0; i < 3; i++) {
  const g = new THREE.Group();
  const flat = c => new THREE.MeshStandardMaterial({ color: c, flatShading: true, roughness: .6 });
  const puck = new THREE.Mesh(new THREE.CylinderGeometry(.9, 1.1, .5, 6), flat('#E4E4E7')); puck.position.y = .25;
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(.06, .06, 1.6, 6), flat('#3F3F46')); mast.position.y = 1.3;
  const beaconMat = new THREE.MeshBasicMaterial({ color: INK.clone() });
  const beacon = new THREE.Mesh(new THREE.OctahedronGeometry(.3, 0), beaconMat); beacon.position.y = 2.25;
  // a thin light column so a 0.8 m node still reads from the wide shot; same on every node until the reveal
  const beamMat = new THREE.MeshBasicMaterial({ color: INK.clone(), transparent: true, opacity: .35, blending: THREE.AdditiveBlending, depthWrite: false });
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(.12, .12, 22, 6, 1, true), beamMat); beam.position.y = 2.25 + 11;
  g.add(puck, mast, beacon, beam); g.scale.setScalar(.35); g.visible = false; scene.add(g);   // about 0.8 m across, beacon 0.8 m up
  for (const m of [puck, mast]) m.castShadow = m.receiveShadow = true;
  const tag = document.createElement('div');
  tag.className = 'tag node-tag'; tag.hidden = true;
  tag.innerHTML = MIC + '<b></b><span></span>';
  document.body.append(tag);
  nodes.push({ g, beaconMat, beamMat, tag, at: new THREE.Vector2(), from: new THREE.Vector3(), relT: Infinity, y: 0 });
}
const nodeUni = uni;                      // the node uniforms live on the ground's uniforms
nodes.forEach((n, i) => { n.at = uni.uNodes.value[i]; });

// ---------- the mark: a green pin dropped from the drone onto the heap ----------
const heapY = heightAt(HEAP.x, HEAP.y);
const pin = new THREE.Group();
{
  const gem = new THREE.Mesh(new THREE.OctahedronGeometry(.7, 0), new THREE.MeshStandardMaterial({ color: GREEN, emissive: GREEN, emissiveIntensity: 1.4, roughness: .3, flatShading: true }));
  gem.scale.set(1, 1.7, 1); gem.position.y = 1.2; pin.add(gem);
  const BEAM = 8;
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(.14, .14, BEAM, 12, 1, true), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uCol: { value: GREEN } },
    vertexShader: `varying float vY; void main(){ vY = position.y / ${BEAM}. + .5; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`,
    fragmentShader: `uniform vec3 uCol; varying float vY; void main(){ gl_FragColor = vec4(uCol * pow(1. - vY, 2.) * .8, 1.); }`,
  }));
  beam.position.y = BEAM / 2; pin.add(beam);
  pin.visible = false; scene.add(pin);
}

// ---------- flight: one spline, low over the rubble, a chase camera that hands over to a slow orbit ----------
// control points sit 12 m over the tallest return within 15 m, so the opener is low but never clips a roof
const lowOver = (x, z, clear = 12) => {
  let h = -Infinity;
  for (let a = 0; a < 8; a++) for (const r of [0, 8, 15]) h = Math.max(h, heightAt(x + Math.cos(a * .785) * r, z + Math.sin(a * .785) * r));
  return h + clear + SKID;
};
const path = new THREE.CatmullRomCurve3([
  [-100, 0, 100], [-70, 0, 62], [-25, 0, 50], [22, 0, 46], [44, 0, 62],
].map(([x, , z]) => new THREE.Vector3(x, lowOver(x, z), z)).concat([new THREE.Vector3(HEAP.x, heapY + HOVER_AGL, HEAP.y)]), false, 'centripetal');
const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const easeOut = t => 1 - Math.pow(1 - t, 3);

const state = { t: 0, listening: 0, listenOn: false, verdict: null, markAt: null, clock: 0, dropReq: null, rotor: 1,
  pile: null, pileVerdicts: null, source: null, rippleAt: null, audioAt: -1, audio: 0, jumped: true };   // no speed from the origin on frame 1
const hoverPt = path.getPoint(1);
const foot = (x, z) => [[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]].map(([a, b]) => heightAt(x + a * SKID * .6, z + b * SKID * .9));
const footY = (x, z) => Math.max(...foot(x, z));
// a flat landing spot on a ring around `c`: high (on the rubble, in view) for one node, low (open ground) for three
function pickLand(c, r0, r1, high, avoid = []) {
  const out = new THREE.Vector2(c.x + r0, c.y);
  for (let best = -Infinity, i = 0; i < 96; i++) {
    const a = i / 32 * Math.PI * 2, d = r0 + (r1 - r0) * (i % 3) / 2, x = c.x + Math.cos(a) * d, z = c.y + Math.sin(a) * d;
    if (Math.hypot(x, z) > 110 || avoid.some(p => Math.hypot(p.x - x, p.y - z) < 14)) continue;
    const f = foot(x, z), top = Math.max(...f), score = (high ? top : -top) - 3 * (top - Math.min(...f));
    if (score > best) { best = score; out.set(x, z); }
  }
  return out;
}
const LAND1 = pickLand(HEAP, 12, 16, true);
// a mission: drone keyframes, node releases, then land and spin the rotors down
let mission = null;
// calls that arrive before the nodes' flight has started (the drone still en route): checked at once, then held, the
// latest of each kind, and applied in order when the flight starts, so a verdict sent early is never lost
const held = new Map();
function hold(name, args, always) {
  if (mission || !(always || state.dropReq)) return false;
  held.delete(name); held.set(name, [...args]); return true;
}
function plan(targets, land, framing) {
  const keys = [{ t: 0, p: (state.from ?? drone.position).clone() }]; state.from = null;
  let t = 0, p = keys[0].p;
  const go = (q, dur) => { if (dur > 0) { t += dur; keys.push({ t, p: q }); p = q; } };
  const cruiseTo = (x, z, clear) => {          // climb, then cross high enough to clear everything on the way
    let h = Math.max(p.y, footY(x, z) + SKID + clear);
    for (let k = 0; k <= 10; k++) h = Math.max(h, heightAt(THREE.MathUtils.lerp(p.x, x, k / 10), THREE.MathUtils.lerp(p.z, z, k / 10)) + SKID + 6);
    if (Math.hypot(x - p.x, z - p.z) < 1) return;
    go(new THREE.Vector3(p.x, h, p.z), (h - p.y) / 5);
    go(new THREE.Vector3(x, h, z), THREE.MathUtils.clamp(Math.hypot(x - p.x, z - p.z) / 10, 2.5, 6));
  };
  targets.forEach((at, i) => {
    cruiseTo(at.x, at.y, 14);
    const low = new THREE.Vector3(at.x, footY(at.x, at.y) + SKID + 2.5, at.y);
    go(low, 2.5);
    Object.assign(nodes[i], { relT: t, from: low.clone(), y: heightAt(at.x, at.y) }); nodes[i].at.copy(at);
    go(low.clone(), .7);                        // hold while the node falls
  });
  cruiseTo(land.x, land.y, 8);
  go(new THREE.Vector3(land.x, footY(land.x, land.y) + SKID, land.y), 2.5);
  const all = [...targets, land], c = all.reduce((s, v) => s.add(v), new THREE.Vector2()).divideScalar(all.length);
  mission = { start: state.clock, keys, count: targets.length, spinAt: t, end: t + 2.6, centre: c, ...framing };
}
// the end bearing for three piles, from the piles alone (never from a verdict): on a wide screen the one that
// spreads them furthest apart side to side, so their tags never touch; on a tall one the one that keeps them inside
// a phone's width (45 m across) yet furthest apart in tag sizes there (a tag is about 36 m across, 23 m deep), since a
// tag with no free spot is hidden. shortcut: piles over 45 m across at every bearing fall back to south, fine for the kit's piles
function bearings(piles) {
  let w = -Infinity, t = -Infinity, Aw = 0, At = 0;
  for (let i = 0; i < 180; i++) {
    const a = i / 180 * Math.PI * 2, x = piles.map(p => p.x * Math.sin(a) - p.y * Math.cos(a)).sort((p, q) => p - q);
    const pref = -.5 * Math.abs(Math.atan2(Math.sin(a - Math.PI / 2), Math.cos(a - Math.PI / 2)));   // from the south when it ties
    if (Math.min(x[1] - x[0], x[2] - x[1]) + pref > w) { w = Math.min(x[1] - x[0], x[2] - x[1]) + pref; Aw = a; }
    const dep = piles.map(p => p.x * Math.cos(a) + p.y * Math.sin(a)), acr = piles.map(p => p.x * Math.sin(a) - p.y * Math.cos(a));
    let m = x[2] - x[0] > 45 ? -1e9 : Infinity;
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) m = Math.min(m, Math.max(Math.abs(acr[i] - acr[j]) / 36, Math.abs(dep[i] - dep[j]) / 23));
    if (m + pref * .1 > t) { t = m + pref * .1; At = a; }
  }
  return { Aw, At };
}
function missionPose(d, out) {                  // returns the rotor speed 0..1
  const k = mission.keys.findIndex(kf => kf.t > d);
  if (k === -1) out.copy(mission.keys.at(-1).p);
  else { const a = mission.keys[k - 1], b = mission.keys[k]; out.lerpVectors(a.p, b.p, ease((d - a.t) / (b.t - a.t))); }
  return d < mission.spinAt ? 1 : Math.pow(1 - Math.min((d - mission.spinAt) / 2.6, 1), 2);
}
const tmp = new THREE.Vector3(), fwd = new THREE.Vector3(), side = new THREE.Vector3(), camPos = new THREE.Vector3(), camLook = new THREE.Vector3();
const prevPos = new THREE.Vector3(); let speed = 0;

function poseAt(t, time, out) {
  const u = THREE.MathUtils.clamp(t / FLIGHT_S, 0, 1), s = ease(u);
  path.getPoint(s, out);
  path.getTangent(Math.min(s, .999), fwd); fwd.y = 0; fwd.normalize();
  out.y += Math.sin(time * 1.3) * .12 + Math.sin(time * .7) * .08;   // hover breath
  return s;
}

const camGoal = new THREE.Vector3(), lookGoal = new THREE.Vector3(), tmp2 = new THREE.Vector2();
// One camera height and angle for every beat (about 38 degrees down from 150-180 m): close up, the 1 m heightfield
// reads as black shards over a blurred photo, so the camera never comes down. From the south while the drone flies
// and listens at the opening spot; for the three piles, a bearing from the piles alone (never from a verdict).
const SOUTH = Math.PI / 2 - .3;
function updateCamera(s, time, md, dt) {
  const d = drone.position, many = mission?.Aw !== undefined, drift = .05 * Math.sin(time * .03);   // a slow sway, bounded: an ever-growing turn swung the phone's stacked piles off screen by the reveal
  const f = many ? mission.centre : !mission && s < 1 ? tmp2.set(d.x, d.z) : mission ? mission.centre : HEAP;
  const a = many ? (innerWidth > innerHeight ? mission.Aw : mission.At) + drift : SOUTH + drift, tall = many && innerWidth < innerHeight, zoom = tall ? 1.9 : 1, R = (many ? 140 : 120) * zoom, Y = (many ? 110 : 95) * zoom;
  const fy = heightAt(f.x, f.y);
  camGoal.set(f.x + Math.cos(a) * R, fy + Y, f.y + Math.sin(a) * R);
  const lift = tall ? 40 : 0;   // aim short of the piles on a phone: they sit in the band between the pill and the panel
  lookGoal.set(f.x + Math.cos(a) * lift, fy + 2, f.y + Math.sin(a) * lift);
  camGoal.y = Math.max(camGoal.y, heightAt(camGoal.x, camGoal.z) + 3);   // never inside a wall
  const k = camPos.lengthSq() ? 1 - Math.exp(-dt * 2) : 1;   // dt is Infinity after a seek: snap
  camPos.lerp(camGoal, k); camLook.lerp(lookGoal, k);
  // a hand-held hover: a few centimetres of slow drift, scaled with distance; the same on every pile
  const sw = camPos.distanceTo(camLook) * .004;
  camera.position.copy(camPos).add(new THREE.Vector3(Math.sin(time * .7) * sw, Math.sin(time * 1.1 + 1) * sw * .5, Math.cos(time * .5) * sw));
  camera.lookAt(camLook);
  fitShadow(camLook, camPos.distanceTo(camLook));
}

const airScale = { value: 1 };   // the air sprites' pixels per metre at 1 m, set in resize()

// ---------- post: MSAA target, soft bloom, ACES, then grain and vignette ----------
const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.addPass(new RenderPass(scene, camera));
const gtao = new GTAOPass(scene, camera, 1, 1);           // contact shadows where rubble meets rubble
gtao.updateGtaoMaterial({ radius: 3, distanceExponent: 1.5, thickness: 2, scale: 1, samples: 12 });
gtao.blendIntensity = .9;
composer.addPass(gtao);
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), .25, .4, 1.3);   // a faint glow on the beacons only; the lit photo stays under
composer.addPass(bloom);
composer.addPass(new OutputPass());
const grade = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uRes: { value: new THREE.Vector2() } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uTime; uniform vec2 uRes; varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){ vec3 c = texture2D(tDiffuse, vUv).rgb;
      vec2 q = (vUv - .5) * vec2(uRes.x / uRes.y, 1.);
      c *= mix(1., .72, smoothstep(.45, 1.1, length(q)));
      c += (hash(vUv * uRes + fract(uTime) * 91.) - .5) * .022;
      gl_FragColor = vec4(c, 1.); }`,
});
composer.addPass(grade);

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false); composer.setSize(w, h);
  bloom.resolution.set(w, h); grade.uniforms.uRes.value.set(w * renderer.getPixelRatio(), h * renderer.getPixelRatio());
  camera.aspect = w / h; camera.updateProjectionMatrix();
  scene.fog.density = w <= 720 ? .0012 : .0022;   // the phone's zoomed tall view looks through more haze: lighter, so the piles read
  airScale.value = h * renderer.getPixelRatio() / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
}
addEventListener('resize', resize); resize();

// ---------- HUD ----------
const $ = id => document.getElementById(id);
const verdictEl = $('verdict');
const VERDICTS = {   // [panel label, panel sub, node callout]; says what was heard, never "survivor confirmed"
  survivor: ['Target rhythm flagged', 'Knocks match the five-tap call-out', 'Target rhythm flagged'],
  debris: ['No target rhythm', 'Knocks read as random settling', 'No target rhythm'],
  unsure: ['Unclear', 'No clear rhythm yet, keep listening', 'Unclear'],
};
const SOURCES = { live: 'the live dish', practice: 'the practice dish (simulated)', simulator: 'the simulator (LIF spiking network)', manual: 'the operator key' };
const PRE = { practice: 'Practice', simulator: 'Simulated' };   // every verdict not from the live dish says so, so a cropped frame never reads as a real hit
const CHIP = { live: 'CL1 dish', practice: 'Practice dish (simulated)', simulator: 'Simulator running live · LIF spiking network', manual: 'Operator key' };   // the panel's source line, set only with a verdict
// the caption while a node listens: the node is a microphone, and the dish is at base, never out on the rubble
const FEED = { live: 'Knocks are sent to the living dish at base.', practice: 'Knocks are sent to the practice dish (simulated) at base.', simulator: 'Knocks are sent to the simulator (LIF spiking network) at base.', manual: 'Knocks are sent to the operator at base.' };
const checkSource = s => { if (!Object.hasOwn(SOURCES, s)) throw new Error(`source must be one of ${Object.keys(SOURCES).join(', ')}`); return s; };
// no percentages anywhere: the reader's score is not the dish recognising the target
function setTag(n, title, sub, cls) {
  if (PRE[state.source] && (cls === 'survivor' || cls === 'flagged' || cls === 'neutral')) sub = `${PRE[state.source]} · ${sub}`;   // on the sub line: the tag keeps its width on a phone
  n.tag.dataset.state = cls; n.tag.querySelector('b').textContent = title; n.tag.querySelector('span').textContent = sub;
}
function renderVerdict() {
  const v = state.verdict, pv = state.pileVerdicts, count = mission?.count ?? 0, on = state.listenOn || count > 0;
  let label, sub, kind;
  if (pv) {
    const f = pv.flagged, k = f.length, hit = pv.target != null && f.includes(pv.target);
    // green only when the caller's truth confirms the flag; no truth given (the spine sends none at chance): a neutral ink flag
    kind = hit ? 'survivor' : !k ? 'debris' : pv.target != null ? 'flagged' : 'neutral';
    label = hit ? VERDICTS.survivor[0] : !k ? 'No target rhythm heard' : pv.target != null ? 'Flagged, no survivor' : `Pile${k > 1 ? 's' : ''} ${f.join(' and ')} flagged`;
    sub = !k ? (pv.list.some(x => x.p_survivor === null) ? 'No pile flagged' : 'All piles: random settling') : kind === 'neutral' ? 'Not confirmed'
      : k === pv.list.length ? 'On every pile' : `Pile${k > 1 ? 's' : ''} ${f.join(' and ')} only`;
  } else if (v) {
    kind = v.kind; [label, sub] = VERDICTS[v.kind];
  } else {
    kind = on ? 'listening' : 'idle';
    label = $('mode-label').textContent;   // the pill's word; hud() keeps the two the same every tick
    sub = count ? FEED[state.feed] ?? 'Knocks are sent to base.' : on ? 'Over the rubble' : 'Listening node stowed';
  }
  const pre = PRE[state.source] ? `${PRE[state.source]} · ` : '';   // a cropped frame must never read as a real hit
  $('tag').children[0].textContent = pre + VERDICTS.survivor[0];
  verdictEl.dataset.state = kind;
  $('v-chip').textContent = v || pv ? CHIP[state.source] : '';
  // desktop: "Practice" on its own small line, so the verdict never wraps; textContent (the pill's copy) still reads "Practice · ..."
  if ((v || pv) && pre) $('v-label').innerHTML = `<span class="pre">${PRE[state.source]}<span class="sep"> · </span></span>${label}`;   // label is our own constant
  else $('v-label').textContent = label;
  $('v-sub').textContent = sub;
  $('v-note').textContent = pv?.note ?? ''; $('v-note').hidden = !pv?.note;   // the caller's caveat, e.g. the reader's held-out score
  document.body.dataset.live = state.source === 'live' || state.feed === 'live' ? '1' : '';   // the brand dot is green only on the live dish
  $('prov').textContent = `Scripted flight. Real 2010 terrain, used as a backdrop only. ${state.source ? `Verdict from ${SOURCES[state.source]}` : 'Awaiting verdict'}.`;
  // node callouts and beacons: identical for every pile until a reveal
  nodes.forEach((n, i) => {
    let col = INK, ring = INK;
    if (pv && pv.list[i].p_survivor === null) setTag(n, 'No score', `Pile ${i + 1}`, 'idle');   // the dish did not score it: ink, neutral
    else if (pv) {
      const fl = pv.flagged.includes(i + 1), s = `Pile ${i + 1}`, T = pv.target;
      const [title, st, c] = T == null ? (fl ? ['Flagged', 'neutral', INK] : ['Not flagged', 'debris', GREY])
        : i + 1 === T ? (fl ? [VERDICTS.survivor[2], 'survivor', GREEN] : ['Missed', 'missed', INK])
        : fl ? ['Flagged, no survivor', 'flagged', AMBER] : [VERDICTS.debris[2], 'debris', GREY];
      setTag(n, title, s, st);
      col = ring = c;
    } else if (v && count === 1) {
      setTag(n, VERDICTS[v.kind][2], 'Listening node', v.kind);
      col = ring = v.kind === 'survivor' ? GREEN : v.kind === 'debris' ? GREY : INK;
    } else if (count > 1) {
      setTag(n, `Pile ${i + 1}`, state.pile === i + 1 ? 'Microphone · listening' : 'Microphone · standing by', 'idle');
    } else setTag(n, 'Listening node', 'Microphone · rotors off, listening', 'idle');
    n.col = col; nodeUni.uNodeCol.value[i].copy(ring);
  });
}

// ---------- the hook for the demo program ----------
window.rescue = {
  // `source` (live, practice or manual) names what the listening feeds, so the panel never says the CL1 dish over the practice one
  listen(on = true, source) { if (source !== undefined) state.feed = checkSource(source); state.listenOn = !!on; renderVerdict(); },
  setVerdict(kind, confidence, source = 'manual') {
    if (!Object.hasOwn(VERDICTS, kind)) throw new Error(`kind must be one of ${Object.keys(VERDICTS).join(', ')}`);
    const c = Number(confidence);
    if (!(c >= 0 && c <= 1)) throw new Error('confidence must be a number in [0, 1]');
    checkSource(source); if (hold('setVerdict', arguments)) return;
    state.source = source; state.verdict = { kind, confidence: c }; state.pileVerdicts = null; state.rippleAt = null;   // a new verdict replaces any pile result, which would otherwise mask it
    if (kind !== 'survivor') { state.markAt = null; pin.visible = false; }   // a verdict that confirms nothing takes the mark down
    if (kind === 'survivor' && mission?.count === 1) state.rippleAt = state.clock;
    renderVerdict();
  },
  markSpot() { if (hold('markSpot', arguments)) return; if (state.markAt === null) { state.markAt = state.clock; state.markFrom = drone.position.clone(); } },
  dropNode(source) { if (source !== undefined) state.feed = checkSource(source); if (state.dropReq && !mission) held.clear(); state.dropReq = [HEAP]; },
  dropNodes(piles = PILES, source) {
    if (!Array.isArray(piles) || piles.length !== 3) throw new Error('dropNodes takes three piles, [x, z] in scene metres');
    if (source !== undefined) state.feed = checkSource(source);
    if (state.dropReq && !mission) held.clear();   // a new round replacing a drop still pending: its held calls belong to the old round
    state.dropReq = piles.map(p => new THREE.Vector2(...(p.isVector2 ? [p.x, p.y] : p)));
    if (mission) { state.from = drone.position.clone(); mission = null; }   // take off from wherever the last node left it
    Object.assign(state, { verdict: null, markAt: null, source: null, rippleAt: null, pile: null, pileVerdicts: null, listenOn: false });
    pin.visible = false;
    for (const n of nodes) { n.g.visible = false; n.tag.hidden = true; n.relT = Infinity; n.col = INK; }
    renderVerdict();
  },
  listenPile(i) {
    const n = mission?.count ?? state.dropReq?.length ?? 3;
    if (!(n > 1) || !(i >= 1 && i <= n)) throw new Error('listenPile(i): i is 1..3, with three piles');
    if (hold('listenPile', arguments, true)) return;
    state.pile = i; renderVerdict();
  },
  // `target` is the pile the survivor is really under, when the caller knows it (a scripted run): only then can a flag turn green
  // `note` is an optional caveat shown under the verdict, e.g. the reader's held-out score against shuffled labels
  setPileVerdicts(list, flagged = [], source = 'manual', { target = null, note } = {}) {
    const n = mission?.count ?? state.dropReq?.length ?? 3;   // before any drop: held until dropNodes() has run
    if (n < 2 || !Array.isArray(list) || list.length !== n) throw new Error(`setPileVerdicts needs ${n > 1 ? n : 3} entries, with three piles`);
    for (const x of list) if (!(x?.p_survivor === null || x?.p_survivor >= 0 && x.p_survivor <= 1)) throw new Error('each entry needs p_survivor in [0, 1], or null when unscored');
    flagged ??= [];
    if (!Array.isArray(flagged) || new Set(flagged).size !== flagged.length || !flagged.every(i => Number.isInteger(i) && i >= 1 && i <= n))
      throw new Error('flagged is a list of distinct pile numbers 1..n (empty: no pile)');
    if (!(target === null || Number.isInteger(target) && target >= 1 && target <= n)) throw new Error('target is a pile number 1..n, or null');
    if (!(note === undefined || typeof note === 'string')) throw new Error('note is a string, or left out');
    checkSource(source); if (hold('setPileVerdicts', arguments, true)) return;
    state.source = source; state.pile = null;
    const ps = list.map(x => x.p_survivor === null ? null : Number(x.p_survivor));
    flagged = flagged.filter(i => ps[i - 1] !== null);   // an unscored pile is never flagged
    state.pileVerdicts = { list: ps.map(p_survivor => ({ p_survivor })), flagged: [...flagged].sort(), target, note };
    state.rippleAt = flagged.includes(target) ? state.clock : null;
    renderVerdict();
  },
  audioLevel(x) { state.audio = THREE.MathUtils.clamp(Number(x) || 0, 0, 1); state.audioAt = state.clock; },
  reset() {
    Object.assign(state, { jumped: true, t: 0, listenOn: false, verdict: null, markAt: null, dropReq: null, rotor: 1,
      pile: null, pileVerdicts: null, source: null, feed: null, rippleAt: null });
    mission = null; held.clear(); state.listening = 0; pin.visible = false; $('tag').hidden = true;
    for (const n of nodes) { n.g.visible = false; n.tag.hidden = true; n.relT = Infinity; }
    renderVerdict();
  },
  look() {   // everything on screen that could hint at a pile's answer, for the no-leak check in scripts/shoot.py
    return { label: $('v-label').textContent, sub: $('v-sub').textContent, note: $('v-note').textContent, card: verdictEl.dataset.state,
      wave: [...Array(64)].map((_, k) => waveSample(k * .05).toFixed(4)), mode: document.body.dataset.mode,
      nodes: nodes.map((n, i) => ({ tag: n.tag.dataset.state, text: n.tag.textContent, hidden: n.tag.hidden, col: (n.col || INK).getHexString(),
        a: uni.uNodeA.value[i].toArray().map(v => v.toFixed(3)), ring: uni.uNodeCol.value[i].getHexString() })) };
  },
  seek(seconds) { state.t = Math.max(0, Number(seconds) || 0); state.jumped = true; },
  get state() {
    const md = mission ? state.clock - mission.start : -1;
    return { t: state.t, arrived: state.t >= FLIGHT_S, listening: state.listenOn, verdict: state.verdict, source: state.source, marked: state.markAt !== null,
      nodes: mission?.count ?? 0, node: !mission ? null : md > mission.end ? 'landed' : 'dropping', pile: state.pile, pileVerdicts: state.pileVerdicts,
      pinVisible: pin.visible, dronePx: dronePx(), droneBox: droneBox(), fps: fps.value, lite: LITE, tagCross };
  },
};
function droneBox() {                        // the drone's screen box in CSS px, for the phone's drone-vs-HUD check
  const b = new THREE.Box3().setFromObject(drone), c = [];
  for (let i = 0; i < 8; i++) c.push(new THREE.Vector3(i & 1 ? b.max.x : b.min.x, i & 2 ? b.max.y : b.min.y, i & 4 ? b.max.z : b.min.z).project(camera));
  const xs = c.map(v => (v.x * .5 + .5) * innerWidth), ys = c.map(v => (-v.y * .5 + .5) * innerHeight);
  return { left: Math.min(...xs), right: Math.max(...xs), top: Math.min(...ys), bottom: Math.max(...ys) };
}
function dronePx() {                          // the drone's on-screen height in CSS px, for the >= 40 px check
  let lo = Infinity, hi = -Infinity; const v = new THREE.Vector3();
  drone.traverse(o => { if (!o.isMesh || o.material.transparent) return;
    const pos = o.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) { v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld).project(camera); lo = Math.min(lo, v.y); hi = Math.max(hi, v.y); } });
  return Math.round((hi - lo) / 2 * innerHeight);
}
addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'l') rescue.listen(!state.listenOn);
  else if (k === 's') rescue.setVerdict('survivor', .91, 'manual');
  else if (k === 'd') rescue.setVerdict('debris', .74, 'manual');
  else if (k === 'm') rescue.markSpot();
  else if (k === 'n') rescue.dropNode();
  else if (k === 'r') rescue.reset();
});
if (params.has('t')) rescue.seek(params.get('t'));
renderVerdict();
document.querySelector('.keys').hidden = !params.has('keys');   // presenter hint row, off on screen by default
new ResizeObserver(([e]) => document.body.style.setProperty('--credit-h', `${e.target.offsetHeight}px`)).observe($('foot'));

// ---------- the verdict card's waveform ----------
// The real input when the demo program feeds rescue.audioLevel(); otherwise rotor hum in flight and a neutral floor.
// Never drawn from a verdict, so it can't tell one pile from another.
const wave = $('wave'), wctx = wave.getContext('2d'), WB = 88, wbuf = new Float32Array(WB);
let wAcc = 0;
const hash = x => { const s = Math.sin(x * 127.1) * 43758.5453; return s - Math.floor(s); };
function waveSample(t) {
  if (state.clock - state.audioAt < .5) return state.audio;
  const a = .03 + hash(t) * .04 + state.rotor * (mission || state.listenOn ? .12 : .3) * (.6 + .4 * hash(t + 3));
  return Math.min(a, 1);
}
function drawWave(dt) {
  for (wAcc += dt; wAcc > 1 / 30; wAcc -= 1 / 30) { wbuf.copyWithin(0, 1); wbuf[WB - 1] = waveSample(state.clock - wAcc); }
  const w = wave.width = wave.clientWidth * devicePixelRatio, h = wave.height = wave.clientHeight * devicePixelRatio, bw = w / WB;
  const col = getComputedStyle(verdictEl).getPropertyValue('--wave') || '#A1A1AA';
  wctx.fillStyle = col;
  for (let i = 0; i < WB; i++) { const bh = Math.max(1, wbuf[i] * h); wctx.fillRect(i * bw, (h - bh) / 2, bw * .6, bh); }
}

// ---------- loop ----------
const fps = { value: 0, frames: 0, since: performance.now() };
// ?lite=1, or on its own when the first 60 frames average under 40 fps: no AO, DPR 1, a smaller shadow map, a third of the air
let LITE = false, autoN = 0, autoT0 = 0;
function setLite() {
  LITE = true; document.body.dataset.lite = '1';
  renderer.setPixelRatio(1); gtao.enabled = false;
  sun.shadow.mapSize.set(1024, 1024); sun.shadow.map?.dispose(); sun.shadow.map = null;
  for (const l of airLayers) l.geometry.setDrawRange(0, Math.ceil(l.geometry.attributes.position.count / 3));
  resize();
}
let last = performance.now();
const ringCol = new THREE.Color();
// ---------- air: low dust haze, two far smoke columns, rotor downwash. All animated on the GPU from uTime ----------
// Soft round sprites sized in metres; quiet greys and dust browns so nothing competes with the node.
const air = { uDown: { value: 0 }, uFoot: { value: 0 } };   // airScale (pixels per metre at 1 m) is set in resize()
function airLayer(n, seedFn, kind) {
  const p = new Float32Array(n * 3), sd = new Float32Array(n);
  for (let i = 0; i < n; i++) { const [x, y, z] = seedFn(i); p.set([x, y, z], i * 3); sd[i] = (i * .618034) % 1; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(p, 3)); g.setAttribute('aSeed', new THREE.BufferAttribute(sd, 1));
  const pts = new THREE.Points(g, new THREE.ShaderMaterial({
    uniforms: { uTime: uni.uTime, uDrone: uni.uDrone, uHalf: uni.uHalf, uScale: airScale, ...air },
    transparent: true, depthWrite: false,
    vertexShader: `uniform float uTime, uHalf, uScale, uDown, uFoot; uniform vec3 uDrone; attribute float aSeed; varying float vA; varying vec3 vCol;
      void main(){ vec3 w = position; float size, a;
        #if ${kind} == 0          // haze: big faint puffs drifting west to east a metre or two up
          w.x = mod(w.x + uTime * .8 + uHalf, 2. * uHalf) - uHalf;
          size = 34. + aSeed * 30.; a = .13 * smoothstep(uHalf, uHalf - 30., length(w.xz)); vCol = vec3(.62, .58, .5);
        #elif ${kind} == 1        // smoke: rises, spreads and leans downwind, fades out by 70 m
          float t = fract(uTime * .025 + aSeed);
          w += vec3(t * t * 30., t * 70., t * 8.);
          size = 3. + t * 22.; a = .5 * smoothstep(0., .08, t) * (1. - t); vCol = mix(vec3(.2, .2, .21), vec3(.55, .55, .56), t);
        #else                     // downwash: dust rolling out from under the rotors when the drone is low
          float t = fract(uTime * .7 + aSeed);
          float ang = aSeed * 51.3;
          w = vec3(uDrone.x + cos(ang) * (.6 + t * 5.), uFoot + .15 + t * .7 * fract(aSeed * 7.), uDrone.z + sin(ang) * (.6 + t * 5.));
          size = .4 + t * 1.8; a = .35 * uDown * (1. - t) * smoothstep(0., .1, t); vCol = vec3(.6, .55, .47);
        #endif
        vec4 mv = modelViewMatrix * vec4(w, 1.); gl_Position = projectionMatrix * mv;
        gl_PointSize = min(size * uScale / -mv.z, 1024.); vA = a; }`,
    fragmentShader: `varying float vA; varying vec3 vCol;
      void main(){ vec2 q = gl_PointCoord - .5; float r = dot(q, q) * 4.; if (r > 1.) discard;
        gl_FragColor = vec4(vCol, vA * (1. - r) * (1. - r)); }`,
  }));
  pts.frustumCulled = false; scene.add(pts); return pts;
}
const rnd = (a => () => (a = (a * 16807) % 2147483647) / 2147483647)(7);
const airLayers = [
  airLayer(70, () => [(rnd() * 2 - 1) * HALF, 1.5 + rnd() * 5, (rnd() * 2 - 1) * HALF], 0),
  // two columns at the far crop edge, away from the piles (west and south-west), base on the ground
  airLayer(90, i => { const [x, z] = i % 2 ? [-128, -52] : [-96, 104]; return [x + rnd() * 2, heightAt(x, z), z + rnd() * 2]; }, 1),
  airLayer(120, () => [0, 0, 0], 2),
];

function frame(now) {
  const dt = Math.min((now - last) / 1000, CAPTURE ? 2.5 : .1); last = now;
  state.clock += dt;
  // listening or dropping before arrival fast-forwards the transit so the drone settles over the heap
  state.t = Math.min(state.t + dt * ((state.listenOn || state.dropReq) && state.t < FLIGHT_S ? 6 : 1), FLIGHT_S + 1e6);
  const time = state.clock;
  prevPos.copy(drone.position);
  const s = poseAt(state.t, time, drone.position);
  const arrived = state.t >= FLIGHT_S;
  if (state.dropReq && arrived && !mission) {
    const many = state.dropReq.length > 1;
    plan(state.dropReq, many ? pickLand(state.dropReq.reduce((a, v) => a.add(v), new THREE.Vector2()).divideScalar(3), 6, 16, false, state.dropReq) : LAND1,
      many ? bearings(state.dropReq) : {});
    state.listenOn = false; renderVerdict();
    const calls = [...held]; held.clear();
    for (const [k, a] of calls) try { rescue[k](...a); } catch (e) { console.error(`held ${k}: ${e.message}`); }
  }
  const md = mission ? state.clock - mission.start : -1;
  state.rotor = mission ? missionPose(md, drone.position) : 1;
  const jumped = state.jumped; state.jumped = false;
  if (!jumped) speed = THREE.MathUtils.lerp(speed, Math.min(prevPos.distanceTo(drone.position) / Math.max(dt, 1e-3), 40), .1);   // a stalled frame is not a 2000 m/s drone
  drone.rotation.y = Math.atan2(-fwd.x, -fwd.z);
  drone.rotation.z = arrived ? 0 : THREE.MathUtils.clamp(-path.getTangent(Math.min(s + .02, 1)).cross(fwd).y * 2, -.25, .25);
  drone.rotation.x = arrived ? 0 : -.12;
  for (const p of props) { p.node.rotateOnAxis(p.axis, dt * 60 * state.rotor); p.disc.material.opacity = .1 * state.rotor; }
  // nodes ride under the drone, fall onto the rubble, then listen; every pile pulses the same way
  const blink = (time % 1.6) < .12 ? 1 : .25;
  nodes.forEach((n, i) => {
    const on = mission && i < mission.count && md >= n.relT;
    n.g.visible = !!on;
    const k = on ? THREE.MathUtils.clamp((md - n.relT) / .5, 0, 1) : 0;
    n.g.position.set(n.at.x, THREE.MathUtils.lerp(n.from.y - SKID, n.y, k * k), n.at.y);
    const down = on && k >= 1, revealed = state.pileVerdicts || (state.verdict && mission?.count === 1);
    const pulse = !down ? 0 : revealed ? (n.col === GREEN ? 1 : 0) : mission.count === 1 || state.pile === i + 1 ? 1 : 0;
    nodeUni.uNodeA.value[i].set(down && revealed ? .8 : 0, pulse,
      down && revealed && n.col === GREEN ? 1 : 0);
    n.beaconMat.color.copy(n.col || INK).multiplyScalar(revealed && n.col !== INK ? 2 : 2 * blink);
    n.beamMat.color.copy(n.col || INK).multiplyScalar(revealed && n.col === GREY ? .25 : 1);
  });

  state.listening = THREE.MathUtils.damp(state.listening, state.listenOn && arrived && !mission ? 1 : 0, 3, dt);
  uni.uListen.value = state.listening;
  ringCol.copy(state.verdict?.kind === 'debris' ? GREY : state.verdict?.kind === 'survivor' ? GREEN : INK);
  uni.uRingCol.value.lerp(ringCol, 1 - Math.exp(-dt * 2));
  arrayRing.material.opacity = state.listening * (.6 + .4 * Math.sin(time * 4));
  arrayRing.material.color.copy(uni.uRingCol.value).multiplyScalar(2.2);
  const agl = drone.position.y - heightAt(drone.position.x, drone.position.z);
  cone.visible = state.listening > .01 && coneEdges(tmp.set(drone.position.x, drone.position.y - .4, drone.position.z), drone.position.y - agl, camera.position);
  cone.material.opacity = .55 * state.listening; cone.material.color.copy(uni.uRingCol.value);
  uni.uDrone.value.copy(drone.position);
  air.uFoot.value = heightAt(drone.position.x, drone.position.z);
  air.uDown.value = state.rotor * THREE.MathUtils.smoothstep(7, 2, agl);   // only when low and the rotors turn
  uni.uTime.value = time; grade.uniforms.uTime.value = time;
  nodeUni.uRippleT.value = state.rippleAt === null ? -1 : time - state.rippleAt;

  if (state.markAt !== null) {
    const m = (state.clock - state.markAt) / 1.3, k = Math.min(m, 1);
    pin.visible = true;
    const y = THREE.MathUtils.lerp(state.markFrom.y - SKID, heapY, k * k);   // falls under gravity
    pin.position.set(THREE.MathUtils.lerp(state.markFrom.x, HEAP.x, easeOut(k)), y, THREE.MathUtils.lerp(state.markFrom.z, HEAP.y, easeOut(k)));
    pin.rotation.y = time * .8;
    pin.children[1].visible = k >= 1;
    uni.uMark.value = k >= 1 ? 1 : 0; uni.uMarkT.value = k >= 1 ? (m - 1) * 1.3 : -1;
    if (k >= 1 && state.rippleAt === null) state.rippleAt = state.clock;
  } else { uni.uMark.value = 0; uni.uMarkT.value = -1; }

  updateCamera(s, time, md, jumped ? Infinity : dt);
  composer.render(dt);
  hud(arrived, md, dt);
  fps.frames++;
  if (!LITE && !CAPTURE && autoN++ < 60) {   // headless capture is always slow, so it never auto-switches
    if (autoN === 1) autoT0 = now; else if (autoN === 60 && 59000 / (now - autoT0) < 40) setLite();
  }
  if (now - fps.since > 1000) { fps.value = fps.frames * 1000 / (now - fps.since); fps.frames = 0; fps.since = now; }
  requestAnimationFrame(frame);
}
let hudTick = 0;
const HUD = ['.brand', '#mode', '#telemetry', '#verdict', '#foot'].map(q => document.querySelector(q));
// Tags in priority order: each takes the first spot beside its anchor (right, left, then a row up or down) that stays
// on screen and touches no HUD box and no tag already placed; the dot stays on the anchor, an elbow joins it to the box.
// No tag covers another tag's anchor, and no leader crosses a tag or another leader; when only a crossing spot is
// left, a pile's tag still takes it (counted in tagCross, which the shoot asserts is 0) and the drone's tag hides.
// A tag with no clear spot is hidden rather than drawn over something.
let tagCross = 0;
const hit = (a, b, m = 0) => a.left < b.right + m && b.left - m < a.right && a.top < b.bottom + m && b.top - m < a.bottom;
function placeTags(list, optional = [], droneKeep = true) {   // a landed drone a few px wide is ground: tags may cover it
  const hudR = HUD.map(e => e.getBoundingClientRect()).filter(r => r.width > 0)   // offsetParent is null for every fixed box
    .map(b => ({ left: b.left - 12, right: b.right + 12, top: b.top - 12, bottom: b.bottom + 12 }));   // 20 px from the HUD
  const tags = [...list.map(t => [...t, false]), ...optional.map(t => [...t, true])].map(([el, p, opt]) => {
    const ax = (p.x * .5 + .5) * innerWidth, ay = (-p.y * .5 + .5) * innerHeight;
    const out = el.hidden || p.z >= 1 || ax < 8 || ax > innerWidth - 8 || ay < 8 || ay > innerHeight - 8;
    return { el, opt, ax, ay, out, w: el.offsetWidth, h: el.offsetHeight };
  });
  const LW = innerWidth <= 720 ? 30 : 14, PR = innerWidth <= 720 ? 18 : 5;   // phone: a longer leader and a wider keep-out, so a label sits off the piles
  const dots = tags.filter(t => !t.out && !t.opt).map(t => [t, { left: t.ax - PR, right: t.ax + PR, top: t.ay - PR, bottom: t.ay + PR }]);
  const solve = order => {   // pure: a spot per tag in this order, and how many piles hide or cross
    const taken = [...hudR, ...droneKeep ? [droneBox()] : []], leads = [], boxes = [], spots = new Map();
    let hidden = 0, cross = 0;
    const clear = r => r.left >= 8 && r.top >= 8 && r.right <= innerWidth - 8 && r.bottom <= innerHeight - 8 && !taken.some(b => hit(r, b, 8));   // 8 px apart
    for (const t of order) {
      if (t.out || t.opt && boxes.some(b => hit(b, { left: t.ax, right: t.ax, top: t.ay, bottom: t.ay }, 6))) continue;   // a covered drone gets no callout
      const { ax, ay, w, h } = t, gap = LW;
      let spot = null, loose = null;
      const cands = [];
      for (const dy of [0, -(h + 8), h + 8, -2 * (h + 8), 2 * (h + 8)]) for (const side of [1, -1]) {
        const left = side > 0 ? ax + gap : ax - gap - w, top = ay - h / 2 + dy, r = { left, top, right: left + w, bottom: top + h };
        const lx = side > 0 ? left : left + w, my = top + h / 2;
        cands.push({ r, side, dy, lead: [{ left: ax - 1, right: ax + 1, top: Math.min(ay, my), bottom: Math.max(ay, my) },   // the elbow's upright, then its arm
          { left: Math.min(ax, lx), right: Math.max(ax, lx), top: my - 1, bottom: my + 1 }] });
      }
      for (const vert of [1, -1]) {   // last resort, for a wide box at mid-screen: straight above or below, a plain upright leader
        const left = THREE.MathUtils.clamp(ax - w / 2, 8, innerWidth - 8 - w), top = vert > 0 ? ay - gap - h : ay + gap, r = { left, top, right: left + w, bottom: top + h };
        if (ax > left + 10 && ax < r.right - 10)
          cands.push({ r, vert, side: 1, dy: 0, lead: [{ left: ax - 1, right: ax + 1, top: vert > 0 ? r.bottom : ay, bottom: vert > 0 ? ay : r.top }] });
      }
      for (const c of cands) {
        const { r, lead } = c;
        if (!clear(r) || hudR.some(b => lead.some(l => hit(l, b)))) continue;
        const crosses = dots.some(([u, d]) => u !== t && hit(r, d)) || leads.some(m => hit(r, m, 2))
          || lead.some(l => boxes.some(b => hit(l, b, 2)) || leads.some(m => hit(l, m, 2)));
        if (!crosses) { spot = c; break; }
        loose ??= c;
      }
      if (!spot && loose && !t.opt) { spot = loose; cross++; }
      if (!spot) { hidden += !t.opt; continue; }
      taken.push(spot.r); boxes.push(spot.r); leads.push(...spot.lead);
      spots.set(t, spot);
    }
    return { spots, score: hidden * 100 + cross, cross };
  };
  // the given order first; when it hides a pile or crosses a leader, the first order of the piles that does neither
  const req = tags.filter(t => !t.opt), opt = tags.filter(t => t.opt);
  let best = solve(tags);
  const perms = a => a.length < 2 ? [a] : a.flatMap((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [x, ...p]));
  if (best.score) for (const o of perms(req)) { const s = solve([...o, ...opt]); if (s.score < best.score) best = s; if (!best.score) break; }
  tagCross = best.cross;
  for (const t of tags) {
    const { el, h } = t, spot = best.spots.get(t);
    if (!spot) { el.hidden = true; continue; }
    el.style.left = `${spot.r.left}px`; el.style.top = `${spot.r.top}px`;
    el.classList.toggle('flip', spot.side < 0); el.style.setProperty('--lw', `${LW}px`);
    el.classList.toggle('vert', !!spot.vert); el.classList.toggle('below', spot.vert < 0); el.style.setProperty('--vx', `${t.ax - spot.r.left}px`);
    const mid = h / 2, at = mid - spot.dy;   // box middle and anchor, in the box's own coordinates
    el.style.setProperty('--at', `${at}px`);
    el.style.setProperty('--lt', `${Math.min(mid, at)}px`); el.style.setProperty('--lh', `${Math.max(Math.abs(spot.dy), 1)}px`);
    el.style.setProperty('--bt', spot.dy < 0 ? '1px' : '0px'); el.style.setProperty('--bb', spot.dy >= 0 ? '1px' : '0px');
    el.style.setProperty('--bv', spot.dy ? '1px' : '0px');
  }
}
function hud(arrived, md, dt) {   // the LED goes green only for a confirmed find
  const placing = mission && md < mission.spinAt - 4, landed = mission && md > mission.end;
  const mode = !mission ? (state.markAt !== null ? ['verdict', 'Hovering'] : state.listenOn && arrived ? ['listening', 'Scanning'] : arrived ? ['hover', 'Hovering'] : ['transit', 'En route'])
    : placing ? ['busy', mission.count > 1 ? 'Placing nodes' : 'Placing node'] : !landed ? ['busy', 'Landing']
    : state.pileVerdicts || state.verdict ? [{ survivor: 'verdict', flagged: 'flagged' }[verdictEl.dataset.state] ?? 'busy', $('v-label').textContent] : state.pile ? ['listening', `Pile ${state.pile} listening`] : ['listening', mission.count > 1 ? 'Nodes listening' : 'Node listening'];
  document.body.dataset.mode = mode[0]; $('mode-label').textContent = mode[1];
  if (!state.verdict && !state.pileVerdicts) $('v-label').textContent = mode[1];   // one status word in the pill and the panel
  const dp = drone.position.clone().project(camera);   // the readout is the drone's: shown only while the drone is
  $('telemetry').hidden = Math.abs(dp.x) > 1 || Math.abs(dp.y) > 1 || dp.z > 1;
  const v = new THREE.Vector3(HEAP.x, heapY + 9.5, HEAP.y).project(camera), tag = $('tag');
  const marked = state.markAt !== null && state.clock - state.markAt > 1.4;
  tag.hidden = !marked;
  for (const n of nodes) n.tag.hidden = !(n.g.visible && md > n.relT + .5);
  const dt_ = $('drone-tag'); dt_.hidden = false;   // the drone is a few px in the high shots: a callout says what it is
  placeTags([...nodes.map(n => [n.tag, n.g.position.clone().add({ x: 0, y: 1.2, z: 0 }).project(camera)]),
    [tag, v]], [[dt_, dp]], !landed);
  drawWave(dt);
  if ((hudTick = (hudTick + 1) % 6) && !CAPTURE) return;   // text readouts at ~10 Hz; every frame in capture, where a frame is ~1 s
  const cs = Math.floor(state.clock); $('clock').textContent = `${String(cs / 60 | 0).padStart(2, '0')}:${String(cs % 60).padStart(2, '0')}`;
  const down = mission && md >= mission.spinAt;   // on the ground: the readout says so exactly, not a lagging estimate
  $('t-alt').textContent = `${down ? '0.0' : Math.max(0, drone.position.y - footY(drone.position.x, drone.position.z) - SKID).toFixed(1)} m AGL`;
  $('t-rot').textContent = state.rotor > 0 ? `${Math.round(state.rotor * 5400)} rpm` : 'stopped';
  $('t-spd').textContent = `${(arrived && !mission || down ? 0 : speed).toFixed(1)} m/s`;
  $('t-hdg').textContent = `${String(Math.round((Math.atan2(fwd.x, -fwd.z) * 180 / Math.PI + 360) % 360)).padStart(3, '0')} deg`;   // no degree sign or lat/lon: the real terrain is a backdrop, never a located find
}
if (params.get('lite') === '1') setLite();
requestAnimationFrame(frame);
