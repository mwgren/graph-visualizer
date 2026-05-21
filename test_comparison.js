//500 steps with nudgeBoth, then 500 steps with TFJS.


const tf = require('@tensorflow/tfjs');

/* ─── Data ─── */
const dataset_one = [
  [14, 20], [16, 28], [18, 35], [20, 42], [22, 51],
  [24, 60], [26, 74], [28, 85], [30, 98], [32, 110]
];

/* ═══════════════════════════════════════════════════
nudgeBoth  500 steps
═══════════════════════════════════════════════════ */

function readTable(rows) {
  const pts = [];
  rows.forEach(row => {
    const x = parseFloat(row[0]);
    const y = parseFloat(row[1]);
    if (!isNaN(x) && !isNaN(y)) pts.push({ x, y });
  });
  return pts;
}

const LEARN_RATE = 0.1;
let dataPoints = readTable(dataset_one);

function meanX(pts) {
  let sum = 0;
  for (let i = 0; i < pts.length; i++) sum += pts[i].x;
  return sum / pts.length;
}
function stdX(pts) {
  const mx = meanX(pts);
  let sumSq = 0;
  for (let i = 0; i < pts.length; i++) sumSq += (pts[i].x - mx) ** 2;
  return Math.sqrt(sumSq / pts.length) || 1;
}

function nudgeBoth(steps = 1) {
  const mx = meanX(dataPoints);
  const sx = stdX(dataPoints);
  for (let step = 0; step < steps; step++) {
    const n = dataPoints.length;
    const aNorm = currentA + currentB * mx;
    const bNorm = currentB * sx;
    let sumErrors = 0;
    let sumWeightedErrors = 0;
    dataPoints.forEach(point => {
      const xNorm       = (point.x - mx) / sx;
      const predicted   = aNorm + bNorm * xNorm;
      const felmarginal = point.y - predicted;
      sumErrors         += felmarginal;
      sumWeightedErrors += xNorm * felmarginal;
    });
    const meanError         = sumErrors         / n;
    const meanWeightedError = sumWeightedErrors / n;
    const aNormNew = aNorm + LEARN_RATE * meanError;
    const bNormNew = bNorm + LEARN_RATE * meanWeightedError;
    currentB = bNormNew / sx;
    currentA = aNormNew - currentB * mx;
  }
}

/* initial values */
let sum = 0;
for (let i = 0; i < dataPoints.length; i++) sum += dataPoints[i].y;
const meanY = sum / dataPoints.length;
let currentA = meanY;
let currentB = 0.01;

/* run 500 steps */
nudgeBoth(500);

console.log('After 500 steps with nudgeBoth:');
console.log(`  a = ${currentA.toFixed(6)}`);
console.log(`  b = ${currentB.toFixed(6)}`);

/* ═══════════════════════════════════════════════════
TFJS 500 steps
═══════════════════════════════════════════════════ */

/* same data, separate x and y, normalize x */
const xsRaw  = dataset_one.map(d => d[0]);
const ys     = dataset_one.map(d => d[1]);
const mx     = xsRaw.reduce((s, x) => s + x, 0) / xsRaw.length;
const sx     = Math.sqrt(xsRaw.reduce((s, x) => s + (x - mx) ** 2, 0) / xsRaw.length);
const xsNorm = xsRaw.map(x => (x - mx) / sx);

/* same initial values, but in normalized space */
let aNorm = tf.variable(tf.scalar(meanY));
let bNorm = tf.variable(tf.scalar(0.01));

const xs  = tf.tensor1d(xsNorm);
const ysT = tf.tensor1d(ys);

const predict = x => aNorm.add(bNorm.mul(x));
const loss    = (pred, label) => pred.sub(label).square().mean();
const optimizer = tf.train.sgd(0.2);   // 2× because TFJS gradient has factor of 2

/* run 500 steps */
for (let i = 0; i < 500; i++) {
  optimizer.minimize(() => loss(predict(xs), ysT));
}

/* convert back to original scale */
const aN = aNorm.dataSync()[0];
const bN = bNorm.dataSync()[0];
const tfB = bN / sx;
const tfA = aN - tfB * mx;

console.log('\nAfter 500 steps with TFJS:');
console.log(`  a = ${tfA.toFixed(6)}`);
console.log(`  b = ${tfB.toFixed(6)}`);

/* cleanup */
xs.dispose(); ysT.dispose(); aNorm.dispose(); bNorm.dispose();