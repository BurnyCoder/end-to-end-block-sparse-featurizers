/* Global: report measured evidence faithfully. Local: Plotly and DOM APIs only.
 * Sources: https://plotly.com/javascript/line-charts/ and MDN DOM documentation. */
const $ = id => document.getElementById(id);
const layout = title => ({title, margin: {t: 45, r: 25, b: 65, l: 75}, paper_bgcolor: '#fff', plot_bgcolor: '#fff'});
const draw = (id, traces, options = {}) => Plotly.react(id, traces, {...layout(''), ...options}, {responsive: true, displaylogo: false});
const option = (select, label, value) => select.add(new Option(label, value));
let evidence;

function pilotCurve() {
  const row = evidence.progress.pilot_trials.find(item => item.name === $('pilot').value);
  draw('pilot-curve', row ? [{type: 'scatter', mode: 'lines+markers', name: row.name,
    x: row.history.map(item => item.images_seen), y: row.history.map(item => item.kl)}] : [],
    {title: 'Recorded pilot training only', xaxis: {title: 'Images seen'}, yaxis: {title: 'Batch KL', rangemode: 'tozero'}});
}

function pilotEvidence() {
  const progress = evidence.progress;
  if (!progress) return;
  $('pilot-evidence').hidden = false;
  $('pilot-status').textContent = `${progress.pilot_completed}/${progress.pilot_planned} completed pilots; ${progress.pilot_started} started. ` +
    `${progress.main_started}/25 main runs started. ${evidence.pilot_observations} Missing endpoints are unmeasured.`;
  for (const row of progress.pilot_trials) {
    const tr = document.createElement('tr');
    [row.cell, row.lr, row.state.status, row.development ? row.development.kl.toPrecision(5) : 'unmeasured',
      row.development ? `${(100 * row.development.accuracy).toFixed(2)}%` : 'unmeasured'].forEach(value => {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    });
    $('pilot-rows').append(tr);
    if (row.history.length) option($('pilot'), `${row.cell}, LR ${row.lr}`, row.name);
  }
  $('pilot').addEventListener('change', pilotCurve);
  pilotCurve();
  const verification = evidence.verification;
  if (verification) {
    $('verification-status').textContent = `Cached/online state difference: ${verification.maximum_state_difference}; ` +
      `largest gradient difference: ${Math.max(...Object.values(verification.maximum_gradient_difference))}. ` +
      'This check fits eight training images and is not a generalization result.';
    draw('verification', [{type: 'scatter', mode: 'lines', x: verification.fixed_batch_kl.map((_, i) => i), y: verification.fixed_batch_kl}],
      {xaxis: {title: 'Optimizer update'}, yaxis: {title: 'Fixed training-batch KL', rangemode: 'tozero'}});
  }
}

function curves() {
  const metric = $('metric').value;
  const labels = [...new Set(evidence.runs.map(row => row.label))];
  draw('curves', labels.map(label => {
    const rows = evidence.runs.filter(row => row.label === label).sort((a, b) => a.k - b.k);
    return {name: label, type: 'scatter', mode: 'lines+markers', x: rows.map(row => 3 * row.k), y: rows.map(row => row[metric]),
      error_y: {type: 'data', symmetric: false, array: rows.map(row => row.image_ci95[metric][1] - row[metric]),
        arrayminus: rows.map(row => row[metric] - row.image_ci95[metric][0])}};
  }), {xaxis: {title: 'Active coordinates per token', type: 'log'}, yaxis: {title: $('metric').selectedOptions[0].text}});
}

function layers() {
  const metric = $('error').value, region = Number($('region').value), k = Number($('k').value);
  draw('layers', evidence.runs.filter(row => row.k === k).map(row => ({name: row.label, type: 'scatter', mode: 'lines+markers',
    x: [6, 7, 8, 9, 10, 11, 12], y: row[metric].map(value => value[region])})),
    {xaxis: {title: 'Raw residual after block'}, yaxis: {title: $('error').selectedOptions[0].text, rangemode: 'tozero'}});
}

function maps(reset = false) {
  const diagnostic = evidence.diagnostics[Number($('condition').value)];
  if (!diagnostic) return;
  if (reset) {
    $('example').replaceChildren(); $('block').replaceChildren();
    diagnostic.examples.forEach((item, i) => option($('example'), `Class ${item.label}: ${item.id}`, i));
    diagnostic.selected_blocks.forEach((item, i) => option($('block'), `Block ${item}`, i));
  }
  const example = diagnostic.examples[Number($('example').value)], block = Number($('block').value);
  draw('map', [{type: 'heatmap', z: example.heatmaps[block], colorscale: 'Viridis', zmin: 0}],
    {title: 'Raw block magnitude on patches', yaxis: {autorange: 'reversed', scaleanchor: 'x'}});
  draw('probabilities', [{name: 'Original', type: 'bar', x: Array.from({length: 10}, (_, i) => i), y: example.original_probabilities},
    {name: 'Inserted', type: 'bar', x: Array.from({length: 10}, (_, i) => i), y: example.modified_probabilities}],
    {barmode: 'group', yaxis: {title: 'Class probability', range: [0, 1]}});
  const names = ['remove_selected', 'remove_random_matched', 'swap_same_class', 'swap_other_class'];
  draw('interventions', [{type: 'bar', x: names, y: names.map(name => diagnostic.interventions.reduce(
    (sum, row) => sum + row.kl[name] - row.kl.baseline, 0) / diagnostic.interventions.length)}],
    {yaxis: {title: 'Mean change in KL', zeroline: true}, title: `${diagnostic.interventions.length} fixed targets; k=16`});
}

async function initialize() {
  const response = await fetch('data.json');
  if (!response.ok) return;
  evidence = await response.json();
  $('status').textContent = `${evidence.label}. ${evidence.narrative}`;
  $('paper').href = 'paper.pdf'; $('paper').textContent = 'Sweep paper ↗';
  pilotEvidence();
  if (!evidence.runs.length) {
    document.querySelectorAll('.main-results').forEach(section => { section.hidden = true; });
    return;
  }
  for (const row of evidence.runs) {
    const tr = document.createElement('tr');
    [row.label, row.k, row.kl.toPrecision(5), `${(100 * row.accuracy).toFixed(2)}%`,
      row.insertion_nmse.toPrecision(5), row.downstream_mse.toPrecision(5)].forEach(value => {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    });
    $('rows').append(tr);
  }
  [...new Set(evidence.runs.map(row => row.k))].sort((a, b) => a - b).forEach(k => option($('k'), k, k));
  $('k').value = '16';
  $('thresholds').textContent = JSON.stringify(evidence.thresholds, null, 2);
  $('pareto').textContent = JSON.stringify(evidence.pareto, null, 2);
  $('metric').addEventListener('change', curves);
  ['k', 'error', 'region'].forEach(id => $(id).addEventListener('change', layers));
  evidence.diagnostics.forEach((item, i) => option($('condition'), item.label, i));
  $('condition').addEventListener('change', () => maps(true));
  ['example', 'block'].forEach(id => $(id).addEventListener('change', () => maps()));
  draw('training', evidence.runs.map(row => ({name: `${row.label}, k=${row.k}`, type: 'scatter', mode: 'lines',
    x: evidence.histories[row.cell].map(item => item.images_seen), y: evidence.histories[row.cell].map(item => item.kl)})),
    {xaxis: {title: 'Images seen'}, yaxis: {title: 'Training-batch KL', rangemode: 'tozero'}});
  curves(); layers(); maps(true);
}

initialize().catch(error => { $('status').textContent = `Evidence could not be loaded: ${error.message}`; });
