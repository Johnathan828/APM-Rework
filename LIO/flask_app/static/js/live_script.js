// Global variable to track if the fetch is automatic or manual
let isAutomaticFetch = false;
let selectedAgentName = null;

function fetchLiveUpdates() {
  let url = '/lives_agents_data';
  if (selectedAgentName) {
    url += `?agent_name=${encodeURIComponent(selectedAgentName)}`;
  }

  fetch(url)
    .then(response => response.json())
    .then(data => {
      const alertTbody = document.getElementById('alert-events-table');
      alertTbody.innerHTML = '';

      // Summary tiles
      document.getElementById('total-events').textContent = data.summary_stats.Total_events ?? 0;
      document.getElementById('average-event-health').textContent = `${data.summary_stats.average_health ?? 0}%`;
      document.getElementById('total-monitored-sections').textContent = data.summary_stats.Total_monitored_sections ?? 0;
      document.getElementById('Total-Tag-Inputs').textContent = data.summary_stats.total_tag_inputs ?? 0;

      // Alerts table
      (data.models || []).forEach(model => {
        if (model.events && model.events.length > 0) {
          model.events.forEach(event => {
            const alertRow = `
              <tr>
                <td><a href="#" onclick="setSelectedAgent('${event.model_name}')">${event.model_name}</a></td>
                <td>${event.model_type ?? ''}</td>
                <td>Triggered</td>
                <td>${event.trigger_time ?? ''}</td>
                <td>${model.probability_date ?? ''}</td>
                <td>${event.score ?? ''}%</td>
                <td>${event.level ?? ''}</td>
              </tr>
            `;
            alertTbody.innerHTML += alertRow;
          });
        }
      });
    })
    .catch(error => console.error('Error fetching data:', error));
}

function setSelectedAgent(agentName) {
  selectedAgentName = agentName;
  isAutomaticFetch = false;
  fetchModelHealth(agentName);
  fetchLiveUpdates();
}

function getRandomColor() {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) color += letters[Math.floor(Math.random() * 16)];
  return color;
}

function fetchModelHealth(agentName) {
  if (!isAutomaticFetch) {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) spinner.style.display = 'block';
  }

  fetch(`/get_model_health?agent_name=${encodeURIComponent(agentName)}`)
    .then(response => response.json())
    .then(data => {
      const plotDiv = document.getElementById('plot-container');
      const healthInfo = document.getElementById('health-info');
      const features = document.getElementById('feature-plots');

      if (plotDiv) plotDiv.innerHTML = '';
      if (healthInfo) healthInfo.innerHTML = '';
      if (features) features.innerHTML = '';

      const spinner = document.getElementById('loading-spinner');
      if (spinner) spinner.style.display = 'none';

      const healthTrace = {
        x: data.timestamps,
        y: data.health_scores,
        mode: 'lines',
        type: 'scatter',
        name: 'Health Score'
      };

      const healthLayout = {
        title: `Health Score for ${agentName}`,
        xaxis: { title: 'Time', type: 'date', tickformat: '%Y-%m-%d %H:%M', hoverformat: '%Y-%m-%d %H:%M', showgrid: true },
        yaxis: { title: 'Health Score', range: [0, 100], showgrid: true },
        autosize: true,
        legend: { orientation: 'h', y: -0.3 }
      };

      Plotly.newPlot(plotDiv, [healthTrace], healthLayout, { responsive: true });

      const plotHeight = 300;
      Object.keys(data.feature_df || {}).forEach((featureName, index) => {
        const featureDiv = document.createElement('div');
        featureDiv.id = `feature-plot-${index}`;
        featureDiv.style.marginBottom = '30px';
        if (features) features.appendChild(featureDiv);

        const featureTrace = {
          x: data.feature_timestamps,
          y: data.feature_df[featureName],
          mode: 'lines',
          type: 'scatter',
          name: featureName,
          line: { color: getRandomColor(), width: 2 }
        };

        const featureLayout = {
          title: `Feature: ${featureName}`,
          xaxis: { title: 'Time', type: 'date', tickformat: '%Y-%m-%d %H:%M', hoverformat: '%Y-%m-%d %H:%M', showgrid: true },
          yaxis: { title: 'Feature Values', showgrid: true },
          autosize: true,
          height: plotHeight
        };

        Plotly.newPlot(featureDiv, [featureTrace], featureLayout, { responsive: true });
      });
    })
    .catch(error => {
      const spinner = document.getElementById('loading-spinner');
      if (spinner) spinner.style.display = 'none';
      console.error('Error fetching model health:', error);
    });
}

function fetchUpdates() {
  fetchLiveUpdates();

  if (selectedAgentName) {
    isAutomaticFetch = true;
    fetchModelHealth(selectedAgentName);
  }
}

fetchUpdates();
setInterval(fetchUpdates, 60000);
window.onload = fetchLiveUpdates;