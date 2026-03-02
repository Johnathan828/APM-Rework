// Javascript top manage the live


// Global variable to track if the fetch is automatic or manual
let isAutomaticFetch = false;
let selectedAgentName = null; // Global variable to track the selected agent

function fetchLiveUpdates() {
    let url = '/lives_agents_data';

    // Append the selectedAgentName as a query parameter if it's set
    if (selectedAgentName) {
        url += `?agent_name=${encodeURIComponent(selectedAgentName)}`;
    }
    
    console.log("Fetching URL: ", url);
    
    // Use the dynamically constructed URL
    fetch(url)
        .then(response => response.json())
        .then(data => {
            // let tbody = document.getElementById('agents-table-body');
            let alertTbody = document.getElementById('alert-events-table');

            // Clear current table rows
            // tbody.innerHTML = '';
            alertTbody.innerHTML = '';

            // Event Summary: Elements for stats
            let totalAnomaliesElem = document.getElementById('total-events');
            let averageHealthElem = document.getElementById('average-event-health');
            let totalMonitoredSectionsElem = document.getElementById('total-monitored-sections');
            let totalTagInputsElem = document.getElementById('Total-Tag-Inputs');

            // Assign Stats
            let totalEvents = data.summary_stats.Total_events;
            let eventAverageHealth = data.summary_stats.average_health;
            let totalMonitoredSections = data.summary_stats.Total_monitored_sections;
            let totalTagInputs = data.summary_stats.total_tag_inputs;

            // Update the HTML values
            totalAnomaliesElem.textContent = totalEvents;
            averageHealthElem.textContent = `${eventAverageHealth}%`;
            totalMonitoredSectionsElem.textContent = totalMonitoredSections;
            totalTagInputsElem.textContent = totalTagInputs;

            // tbody.innerHTML = ''; // Clear current table rows
            alertTbody.innerHTML = '';  // Clear current table rows for alerts

            // let totalAnomalies = 0; // Initialize total anomalies
            // let totalAlertProbability = 0; // To calculate average health
            // let runningModels = data.models.length; // Number of agents/models

            data.models.forEach(model => {
            //     // Add the agent rows to the left-column table
            //     let row = `
            //     <tr>
            //         <td><a href="#" onclick="setSelectedAgent('${model.agent_name}')">${model.agent_name}</a></td>
            //         <td>${model.type}</td>
            //         <td>${model.latest_result}</td>
            //         <td>${model.probability_date}</td>
            //         <td style="display: flex; align-items: center;">
            //             <div class="probability-bar-container" style="width: 80px; height: 25px; margin-right: 10px;">
            //                 <div class="probability-bar-inner"
            //                     style="width: ${model.alert_probability}%; 
            //                             height: 100%;
            //                             background-color: ${model.alert_probability > 70 ? 'green' : model.alert_probability > 50 ? 'yellow' : 'red'};">
            //                 </div>
            //             </div>
            //             <span style="font-weight: bold;">${model.alert_probability}%</span>
            //         </td>
            //     </tr>
            //     `;
            //     tbody.innerHTML += row;

            //     // Count anomalies where alert probability is less than 67
            //     if (model.alert_probability < 67) {
            //         totalAnomalies++;
            //     }

            //     // Accumulate total alert probability for average calculation
            //     totalAlertProbability += model.alert_probability;

                // Check for events and add them to the alert-events table
                if (model.events && model.events.length > 0) {
                    model.events.forEach(event => {
                        let alertRow = `
                        <tr>
                            <td><a href="#" onclick="setSelectedAgent('${event.model_name}')">${event.model_name}</a></td>
                            <td>${event.model_type}</td>
                            <td>Triggered</td>
                            <td>${event.trigger_time}</td>
                            <td>${model.probability_date}</td>
                            <td>${event.score}%</td>
                            <td>${event.level}</td>
                        </tr>
                        `;
                        alertTbody.innerHTML += alertRow;
                    });
                }
            });



            // Calculate average health as the average of alert probabilities
            let averageHealth = Math.round(totalAlertProbability / runningModels);

            // Update tiles with the calculated values
            updateTiles(totalAnomalies, averageHealth, runningModels);
        })
        .catch(error => console.error('Error fetching data:', error));
}


// Function to set the selected agent and fetch its health data
function setSelectedAgent(agentName) {
    console.log("Selected Agent: ", agentName); // Debug log
    
    selectedAgentName = agentName; // Update the selected agent name globally
    isAutomaticFetch = false; // Mark this as a manual fetch
    fetchModelHealth(agentName); // Fetch the model health immediately when the agent is selected
    fetchLiveUpdates() // Also Update Tables of model selection
}

// Function to generate random color in HEX format
function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}


// Function to fetch health details of the selected agent and plot it using Plotly.js
function fetchModelHealth(agentName) {
    // Show the loading spinner only if it's a manual fetch
    if (!isAutomaticFetch) {
        document.getElementById('loading-spinner').style.display = 'block';
    }

    fetch(`/get_model_health?agent_name=${encodeURIComponent(agentName)}`)
        .then(response => response.json())
        .then(data => {
            // At this point, the data is ready, now we can clear the previous content
            let plotDiv = document.getElementById('plot-container');
            let healthInfo = document.getElementById('health-info');
            let features = document.getElementById('feature-plots');

            // Clear the old plot and info only when the new data is ready to be plotted
            plotDiv.innerHTML = ''; 
            healthInfo.innerHTML = '';
            features.innerHTML = '';

            // Hide the loading spinner when data is loaded
            document.getElementById('loading-spinner').style.display = 'none';

            // Prepare data for Plotly
            const healthTrace = {
                x: data.timestamps, // Timestamps on the X-axis (formatted as dates)
                y: data.health_scores, // Health scores on the Y-axis
                mode: 'lines',
                type: 'scatter',
                name: 'Health Score'
            };

            // Plot layout with responsive sizing
            const healthLayout = {
                title: `Health Score for ${agentName}`,
                xaxis: {
                    title: 'Time',
                    type: 'date',
                    tickformat: '%Y-%m-%d %H:%M',
                    hoverformat: '%Y-%m-%d %H:%M',
                    showgrid: true,
                },
                yaxis: {
                    title: 'Health Score',
                    range: [0, 100],
                    showgrid: true
                },
                autosize: true,
                legend: {
                    orientation: 'h',  // Horizontal legend layout
                    y: -0.3  // Positioning the legend below the plot, adjust y to control distance
                }
            };

            const config = { responsive: true };

            // Render the health score plot using Plotly.js
            Plotly.newPlot(plotDiv, [healthTrace], healthLayout, config);

            // Plot each feature in its own plot ############################################# Features
            const plotHeight = 300;
        
            Object.keys(data.feature_df).forEach((featureName, index) => {
                // Create a new div for each feature plot
                const featureDiv = document.createElement('div');
                featureDiv.id = `feature-plot-${index}`;
                featureDiv.style.marginBottom = '30px'; // Add some space between plots
                features.appendChild(featureDiv);  // Append the new div to the features container

                // Prepare the trace for the current feature
                const featureTrace = {
                    x: data.feature_timestamps,  // Use the timestamps for the x-axis
                    y: data.feature_df[featureName],  // Feature values for each column
                    mode: 'lines',
                    type: 'scatter',
                    name: featureName,  // Column name as the trace name
                    line: {
                        color: getRandomColor(),  // Assign a random color
                        width: 2  // Line width
                    }
                };

                // Layout for the individual feature plot
                const featureLayout = {
                    title: `Feature: ${featureName}`,  // Add the feature name in the title
                    xaxis: {
                        title: 'Time',
                        type: 'date',
                        tickformat: '%Y-%m-%d %H:%M',
                        hoverformat: '%Y-%m-%d %H:%M',
                        showgrid: true,
                    },
                    yaxis: {
                        title: 'Feature Values',
                        showgrid: true
                    },
                    autosize: true,
                    height: plotHeight // Set the height for the plot
                };

                // Render the individual plot in the newly created div
                Plotly.newPlot(featureDiv, [featureTrace], featureLayout, config);
            });
        })
        .catch(error => {
            // Hide the loading spinner if there’s an error
            document.getElementById('loading-spinner').style.display = 'none';
            console.error('Error fetching model health:', error);
        });
}


// Function to update the tile values dynamically
function updateTiles(totalAnomalies, averageHealth, anotherTotal) {
    document.getElementById('total-anomalies').textContent = totalAnomalies;
    document.getElementById('average-health').textContent = `${averageHealth}%`;
    document.getElementById('another-total').textContent = anotherTotal;
}

// Run the function immediately and then every 60 seconds
function fetchUpdates() {
    fetchLiveUpdates();

    // If an agent has been selected, fetch its health data automatically
    if (selectedAgentName) {
        isAutomaticFetch = true; // Mark this as an automatic fetch
        fetchModelHealth(selectedAgentName);
    }
}

// Run the function immediately when the page loads
fetchUpdates();

// Set the interval to run the function every 60 seconds thereafter
setInterval(fetchUpdates, 60000);


// Fetch immediately on page load
window.onload = fetchLiveUpdates;
