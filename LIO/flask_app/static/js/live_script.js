// flask_app/static/js/live_script.js

let isAutomaticFetch = false;
let selectedAgentName = null;

function base() {
  // Must be injected by template as window.baseUrl, e.g. "/Neuromine/LIO/APM/"
  return window.baseUrl || "/";
}

// Convert "YYYY-MM-DD HH:MM:SS" -> "YYYY-MM-DDTHH:MM:SS"
// Helps Plotly parse reliably. If already ISO, return as-is.
function toIsoish(ts) {
  if (!ts) return ts;
  if (typeof ts !== "string") return ts;
  if (ts.includes("T")) return ts;
  // only replace first space (date/time separator)
  return ts.replace(" ", "T");
}

function getRandomColor() {
  const letters = "0123456789ABCDEF";
  let color = "#";
  for (let i = 0; i < 6; i++) color += letters[Math.floor(Math.random() * 16)];
  return color;
}

function updateTiles(totalTriggered, averageHealth, totalModels, offSections, normalSections) {
  document.getElementById("total-anomalies").textContent = totalTriggered;
  document.getElementById("average-health").textContent = `${averageHealth}%`;
  document.getElementById("another-total").textContent = totalModels;
  document.getElementById("latest-results-on").textContent = normalSections;
  document.getElementById("latest-results-off").textContent = offSections;
}

function setSelectedAgent(agentName, event) {
  if (event) event.preventDefault();
  selectedAgentName = agentName;
  isAutomaticFetch = false;
  fetchModelHealth(agentName);
  fetchLiveUpdates();
}

function fetchLiveUpdates() {
  let url = base() + "lives_agents_data";
  if (selectedAgentName) {
    url += `?agent_name=${encodeURIComponent(selectedAgentName)}`;
  }

  fetch(url)
    .then((r) => r.json())
    .then((data) => {
      const tbody = document.getElementById("agents-table-body");
      const alertTbody = document.getElementById("alert-events-table");

      tbody.innerHTML = "";
      alertTbody.innerHTML = "";

      // Event summary tiles
      document.getElementById("total-events").textContent = data.summary_stats?.Total_events ?? 0;
      document.getElementById("average-event-health").textContent = `${data.summary_stats?.average_health ?? 0}%`;
      document.getElementById("total-monitored-sections").textContent = data.summary_stats?.Total_monitored_sections ?? 0;
      document.getElementById("Total-Tag-Inputs").textContent = data.summary_stats?.total_tag_inputs ?? 0;

      const models = data.models || [];

      let totalTriggered = 0;
      let off_sections = 0;
      let normal_sections = 0;

      // This remains "Total Running Agents" in your UI (it’s really "total models returned")
      // If you later want it to show only ON sensors, we can change it similarly.
      const totalModels = models.length;

      // Group by plant_section (legacy)
      const groups = models.reduce((acc, m) => {
        const sec = m.plant_section || "Unknown";
        acc[sec] = acc[sec] || [];
        acc[sec].push(m);
        return acc;
      }, {});

      Object.keys(groups).forEach((sec) => {
        groups[sec].sort((a, b) => (a.alert_probability ?? 0) - (b.alert_probability ?? 0));
      });

      Object.keys(groups)
        .sort()
        .forEach((plantSection) => {
          const safeName = plantSection.replace(/\s+/g, "_");

          const groupHeader = `
            <tr>
              <td colspan="6" style="font-weight:bold; background:#b2babb; text-align:center; font-size:1.5rem;">
                ${plantSection}
              </td>
            </tr>
          `;
          tbody.innerHTML += groupHeader;

          const rows = groups[plantSection];

          rows.forEach((model, index) => {
            const imageUrl = `${base()}get-image/${safeName}.gif`;
            const imageCell =
              index === 0
                ? `<td rowspan="${rows.length}" style="text-align:center; vertical-align:middle; background:white;">
                     <img src="${imageUrl}" alt="${plantSection}" style="height:150px;">
                   </td>`
                : "";

            const barColor =
              model.latest_result === "No Data" || model.latest_result === "Section Off"
                ? "grey"
                : model.latest_result === "Normal"
                ? "green"
                : model.latest_result === "Warning"
                ? "yellow"
                : "red";

            const row = `
              <tr>
                ${imageCell}
                <td><a href="#" onclick="setSelectedAgent('${model.agent_name}', event)">${model.agent_name}</a></td>
                <td>${model.type ?? ""}</td>
                <td class="lates_results_td" data-status="${model.latest_result}">${model.latest_result}</td>
                <td>${model.probability_date ?? ""}</td>
                <td style="display:flex; align-items:center;">
                  <div class="probability-bar-container" style="width:80px; height:25px; margin-right:10px;">
                    <div class="probability-bar-inner" style="width:${model.alert_probability ?? 0}%; height:100%; background-color:${barColor};"></div>
                  </div>
                  <span style="font-weight:bold;">${model.alert_probability ?? 0}%</span>
                </td>
              </tr>
            `;
            tbody.innerHTML += row;

            if ((model.alert_probability ?? 100) < 67 && model.latest_result === "Triggered") totalTriggered++;
            if (model.latest_result === "Section Off") off_sections++;
            if (model.latest_result === "Normal") normal_sections++;

            // Alerts table
            if (model.events && model.events.length > 0) {
              model.events.forEach((ev) => {
                const alertRow = `
                  <tr>
                    <td><a href="#" onclick="getAlertDetail('${ev.model_name}', '${ev.trigger_time}', event)">${ev.model_name}</a></td>
                    <td>${ev.model_type ?? ""}</td>
                    <td>Triggered</td>
                    <td>${ev.trigger_time ?? ""}</td>
                    <td>${model.probability_date ?? ""}</td>
                    <td>${(ev.score ?? "")}%</td>
                    <td>${ev.level ?? ""}</td>
                  </tr>
                `;
                alertTbody.innerHTML += alertRow;
              });
            }
          });
        });

      // ✅ FIX: Use backend-computed overall average that EXCLUDES "Section Off"
      const averageHealth = Math.round(data.summary_stats?.overall_average_health ?? 0);

      updateTiles(totalTriggered, averageHealth, totalModels, off_sections, normal_sections);
    })
    .catch((err) => console.error("fetchLiveUpdates error:", err));
}

function fetchModelHealth(agentName) {
  if (!isAutomaticFetch) {
    const spinner = document.getElementById("loading-spinner");
    if (spinner) spinner.style.display = "block";
  }

  fetch(`${base()}get_model_health?agent_name=${encodeURIComponent(agentName)}`)
    .then((r) => r.json())
    .then((data) => {
      const plotDiv = document.getElementById("plot-container");
      const healthInfo = document.getElementById("health-info");
      const features = document.getElementById("feature-plots");

      if (plotDiv) plotDiv.innerHTML = "";
      if (healthInfo) healthInfo.innerHTML = "";
      if (features) features.innerHTML = "";

      const spinner = document.getElementById("loading-spinner");
      if (spinner) spinner.style.display = "none";

      const xHealth = (data.timestamps || []).map(toIsoish);

      const healthTrace = {
        x: xHealth,
        y: data.health_scores || [],
        mode: "lines",
        type: "scatter",
        name: "Health Score",
      };

      const healthLayout = {
        title: `Health Score for ${agentName}`,
        xaxis: {
          title: "Time",
          type: "date",
          tickformat: "%Y-%m-%d %H:%M",
          hoverformat: "%Y-%m-%d %H:%M",
          showgrid: true,
        },
        yaxis: { title: "Health Score", range: [0, 100], showgrid: true },
        autosize: true,
        legend: { orientation: "h", y: -0.3 },
      };

      Plotly.newPlot(plotDiv, [healthTrace], healthLayout, { responsive: true });

      const plotHeight = 300;
      const featureDf = data.feature_df || {};
      const xFeat = (data.feature_timestamps || []).map(toIsoish);

      Object.keys(featureDf).forEach((featureName, index) => {
        const featureDiv = document.createElement("div");
        featureDiv.id = `feature-plot-${index}`;
        featureDiv.style.marginBottom = "30px";
        if (features) features.appendChild(featureDiv);

        const featureTrace = {
          x: xFeat,
          y: featureDf[featureName] || [],
          mode: "lines",
          type: "scatter",
          name: featureName,
          line: { color: getRandomColor(), width: 2 },
        };

        const details = data.feature_details || {};
        const featDetails = details.Features && details.Features[featureName] ? details.Features[featureName] : null;

        const featureLayout = {
          title: {
            text: featDetails ? featDetails.description : featureName,
            font: { size: 12 },
          },
          xaxis: {
            title: "Time",
            type: "date",
            tickformat: "%Y-%m-%d %H:%M",
            hoverformat: "%Y-%m-%d %H:%M",
            showgrid: true,
          },
          yaxis: featDetails
            ? { title: `${featDetails.unit_description} (${featDetails.unit})`, showgrid: true }
            : { title: "Feature Values", showgrid: true },
          autosize: true,
          height: plotHeight,
        };

        if (data.feature_state && (data.feature_state[featureName] || 0) > 0) {
          featureLayout.plot_bgcolor = "rgba(255, 18, 47, 0.2)";
        }

        Plotly.newPlot(featureDiv, [featureTrace], featureLayout, { responsive: true });
      });
    })
    .catch((err) => {
      const spinner = document.getElementById("loading-spinner");
      if (spinner) spinner.style.display = "none";
      console.error("fetchModelHealth error:", err);
    });
}

function plotModelHealth(modelHealthData, container, model_details) {
  // If timestamp is already a display string, Date parsing can be risky.
  // Prefer ISO-ish conversion if possible; fallback to Date(...) anyway.
  const xValues = (modelHealthData || []).map((entry) => {
    const ts = entry.timestamp;
    if (!ts) return null;

    // Try "YYYY-MM-DD HH:MM:SS" first
    if (typeof ts === "string" && ts.length >= 19) {
      const isoish = toIsoish(ts.substring(0, 19));
      const d = new Date(isoish);
      if (!isNaN(d.getTime())) return d.toISOString();
    }

    // fallback
    const d2 = new Date(ts);
    return isNaN(d2.getTime()) ? null : d2.toISOString();
  });

  const modelKey =
    modelHealthData && modelHealthData[0] ? Object.keys(modelHealthData[0]).find((k) => k !== "timestamp") : null;

  const yValues = modelKey ? modelHealthData.map((entry) => entry[modelKey]) : [];

  const trace = { x: xValues, y: yValues, type: "scatter" };
  const layout = {
    title: model_details && model_details.Model_Description ? model_details.Model_Description : "Model Health",
    xaxis: { type: "date" },
  };
  Plotly.newPlot(container, [trace], layout, { responsive: true });
}

function plotFeatureContribution(featureData, container, model_details, isHighlighted) {
  const xValues = (featureData || []).map((entry) => {
    const ts = entry.timestamp;
    if (!ts) return null;

    if (typeof ts === "string" && ts.length >= 19) {
      const isoish = toIsoish(ts.substring(0, 19));
      const d = new Date(isoish);
      if (!isNaN(d.getTime())) return d.toISOString();
    }

    const d2 = new Date(ts);
    return isNaN(d2.getTime()) ? null : d2.toISOString();
  });

  const yValues = (featureData || []).map((entry) => entry.value);

  const title =
    typeof model_details === "string"
      ? model_details
      : model_details && model_details.description
      ? model_details.description
      : "Feature";

  const layout = {
    title: { text: title, font: { size: 12 } },
    xaxis: { title: "Date", type: "date", tickformat: "%Y-%m-%d %H:%M", hoverformat: "%Y-%m-%d %H:%M", showgrid: true },
    yaxis:
      typeof model_details === "object" && model_details && model_details.unit_description
        ? { title: `${model_details.unit_description} (${model_details.unit})`, showgrid: true }
        : { showgrid: true },
    plot_bgcolor: isHighlighted ? "#ffefef" : "white",
    autosize: true,
    height: 300,
  };

  const trace = { x: xValues, y: yValues, type: "scatter", line: { color: getRandomColor() } };
  Plotly.newPlot(container, [trace], layout, { responsive: true });
}

function getAlertDetail(model_name, trigger_time, event) {
  if (event) event.preventDefault();

  let url = base() + "get_alert_detail";
  url += `?agent_name=${encodeURIComponent(model_name)}`;
  url += `&trigger_time=${encodeURIComponent(trigger_time)}`;

  const loadingSpinner = document.getElementById("loading-spinner-alerts");
  if (loadingSpinner) loadingSpinner.style.display = "block";

  fetch(url)
    .then((r) => r.json())
    .then((data) => {
      if (loadingSpinner) loadingSpinner.style.display = "none";
      if (data.error) {
        console.error("getAlertDetail error:", data.error);
        return;
      }

      const tableContainer = document.getElementById("triggered-tags-table-container");
      tableContainer.innerHTML = "";

      const tagsTable = document.createElement("table");
      tagsTable.insertAdjacentHTML("afterbegin", "<tr><th>Tag</th><th>Tag description</th><th>Contribution %</th></tr>");

      (data.triggered_tags || []).forEach((tag) => {
        const row = document.createElement("tr");
        const feat = tag.feature;
        const featDetails =
          data.model_details && data.model_details.Features && data.model_details.Features[feat]
            ? data.model_details.Features[feat]
            : null;
        const tagName = featDetails ? featDetails.tag : feat;
        const desc = featDetails ? featDetails.description : feat;
        row.innerHTML = `<td>${tagName}</td><td>${desc}</td><td>${((tag.value || 0) * 100).toFixed(2)}%</td>`;
        tagsTable.appendChild(row);
      });

      tableContainer.appendChild(tagsTable);

      const healthPlotDiv = document.getElementById("alert_event_overview_plot");
      plotModelHealth(data.model_health_data || [], healthPlotDiv, data.model_details || {});

      const featurePlotsDiv = document.getElementById("alert_event_overview_feature_plots");
      featurePlotsDiv.innerHTML = "";

      const triggered = new Set((data.triggered_tags || []).map((t) => t.feature));
      const contribs = data.feature_contributions || {};

      Object.keys(contribs).forEach((feature) => {
        const plotContainer = document.createElement("div");
        featurePlotsDiv.appendChild(plotContainer);

        const featureDic =
          data.model_details && data.model_details.Features && data.model_details.Features[feature]
            ? data.model_details.Features[feature]
            : data.model_details && data.model_details.filter_tag
            ? data.model_details.filter_tag
            : feature;

        plotFeatureContribution(contribs[feature], plotContainer, featureDic, triggered.has(feature));
      });
    })
    .catch((err) => {
      if (loadingSpinner) loadingSpinner.style.display = "none";
      console.error("getAlertDetail fetch error:", err);
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
setInterval(fetchUpdates, 300000);
window.onload = fetchLiveUpdates;