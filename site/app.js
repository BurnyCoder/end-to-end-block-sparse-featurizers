// Global: measured evidence only. Local: Plotly selections update static result panels.
// Plotly API: https://plotly.com/javascript/plotlyjs-function-reference/.
const $ = (id) => document.getElementById(id); // One scoped DOM helper.
const theme = {paper_bgcolor: '#fff', plot_bgcolor: '#fff', font: {family: 'system-ui', color: '#17313b'}, margin: {l: 65, r: 25, t: 30, b: 75}};
const options = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']};
let evidence; // Immutable downloaded measurements; controls only change their display.
function plot(id, traces, layout = {}) { Plotly.react(id, traces, {...theme, ...layout}, options); }
function populate(id, values) { $(id).replaceChildren(...values.map(([value, text]) => { const item = document.createElement('option'); item.value = value; item.textContent = text; return item; })); }
function comparison() {
  const metric = $('metric').value, rows = evidence.summaries;
  const error = metric === 'kl' ? rows.map(r => r.kl_seed_ci95 ? r.kl_seed_ci95[1] - r.kl : 0) : null;
  plot('comparison', [{type: 'bar', x: rows.map(r => r.label), y: rows.map(r => r[metric]), marker: {color: rows.map(r => r.condition.includes('kl') ? '#157b72' : '#bc693a')}, error_y: error ? {type: 'data', array: error, visible: true} : undefined}], {yaxis: {title: {text: metric === 'kl' ? 'Forward KL (nats / image)' : metric}, rangemode: 'tozero'}});
}
function maps(reset = false) {
  const diagnostic = evidence.diagnostics[Number($('condition').value)];
  if (!diagnostic) return;
  if (reset) populate('example', diagnostic.examples.map((e, i) => [i, `Class ${e.label} · fixed example ${i + 1}`]));
  const example = diagnostic.examples[Number($('example').value)];
  populate('block', example.block_ids.map((id, i) => [i, `Block ${id} · rank ${i + 1}`]));
  showMap();
  plot('probabilities', [{type: 'bar', name: 'Original', x: [...Array(10).keys()], y: example.original_probabilities, marker: {color: '#17313b'}}, {type: 'bar', name: 'BSF inserted', x: [...Array(10).keys()], y: example.modified_probabilities, marker: {color: '#157b72'}}], {barmode: 'group', yaxis: {title: {text: 'Class probability'}, range: [0, 1]}, xaxis: {title: {text: 'Imagenette class index'}}, legend: {orientation: 'h'}});
  $('example-id').textContent = `${example.split} image: ${example.id}. Original photographs are not redistributed.`;
  const names = ['remove_selected', 'remove_random_matched', 'swap_same_class', 'swap_other_class'];
  const rows = diagnostic.interventions;
  plot('interventions', [{type: 'bar', x: names, y: names.map(name => rows.length ? rows.reduce((sum, row) => sum + row.kl[name] - row.kl.baseline, 0) / rows.length : null), marker: {color: '#bc693a'}}], {yaxis: {title: {text: 'Mean change in KL (nats / image)'}, zeroline: true}});
}
function showMap() {
  const d = evidence.diagnostics[Number($('condition').value)], e = d.examples[Number($('example').value)];
  const z = e.heatmaps[Number($('block').value)];
  plot('map', [{type: 'heatmap', z, colorscale: 'Viridis', colorbar: {title: {text: 'Block norm'}}, hovertemplate: 'Patch (%{x}, %{y})<br>Norm %{z:.4g}<extra></extra>'}], {xaxis: {title: {text: 'Patch column'}, constrain: 'domain'}, yaxis: {title: {text: 'Patch row'}, autorange: 'reversed', scaleanchor: 'x'}});
}
async function start() {
  const response = await fetch('data.json');
  if (!response.ok) throw new Error('Measured result file is not available yet.');
  evidence = await response.json();
  $('status').textContent = evidence.label;
  $('finding').textContent = evidence.narrative;
  const facts = [[evidence.profile, 'Experiment profile'], [evidence.runs.length, 'Completed condition / seed evaluations'], [evidence.summaries[0]?.images_per_seed ?? 'Pending', 'Test images per run'], [evidence.config.groups, 'Blocks in the primary dictionary']];
  $('stats').replaceChildren(...facts.map(([value, label]) => { const el = document.createElement('div'); el.className = 'stat'; const number = document.createElement('strong'); number.textContent = value; const caption = document.createElement('span'); caption.textContent = label; el.append(number, caption); return el; }));
  for (const row of evidence.summaries) { const tr = document.createElement('tr'); for (const value of [row.label, row.n_seeds, row.kl.toPrecision(5), `${(100 * row.accuracy).toFixed(2)}%`, row.nmse.toPrecision(5), row.active_blocks.toFixed(2)]) { const td = document.createElement('td'); td.textContent = value; tr.append(td); } $('results').tBodies[0].append(tr); }
  comparison();
  plot('training', Object.entries(evidence.histories).map(([name, rows]) => ({type: 'scatter', mode: 'lines', name, x: rows.map(r => r.images_seen), y: rows.map(r => r.kl), line: {width: 1.5}})), {xaxis: {title: {text: 'Training images seen'}}, yaxis: {title: {text: 'Batch KL (nats / image)'}, rangemode: 'tozero'}, legend: {orientation: 'h'}});
  populate('condition', evidence.diagnostics.map((d, i) => [i, `${d.condition} · seed ${d.seed}`]));
  maps(true);
  $('metric').addEventListener('change', comparison);
  $('condition').addEventListener('change', () => maps(true));
  $('example').addEventListener('change', () => maps(false));
  $('block').addEventListener('change', showMap);
}
start().catch(error => { $('status').textContent = error.message; }); // Honest unavailable state; no mock data.
