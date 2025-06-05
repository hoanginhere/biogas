  // Load content into main area
  function loadPage(page) {
    const allPages = document.querySelectorAll('.page-content');
    allPages.forEach(p => p.style.display = 'none');

    const activePage = document.getElementById(page);
    if (activePage) {
      activePage.style.display = 'block';
    }

    const allLinks = document.querySelectorAll('.nav-link');
    allLinks.forEach(link => link.classList.remove('active'));

    const activeLink = document.querySelector(`a[href='#${page}']`);
    if (activeLink) {
      activeLink.classList.add('active');
    }

    if (activeLink.closest('.submenu')) {
      const parentLink = activeLink.closest('.submenu').parentElement.querySelector('.nav-link');
      if (parentLink) {
        parentLink.classList.add('active');
      }
    }
  }

  // User avatar dropdown menu
  function toggleUserDropdown(event) {
    event.stopPropagation();
    const dropdown = document.querySelector('.user-dropdown');
    dropdown.classList.toggle('show');
  }

  document.addEventListener('click', function () {
    const dropdown = document.querySelector('.user-dropdown');
    if (dropdown.classList.contains('show')) {
      dropdown.classList.remove('show');
    }
  });

  function toggleSubmenu(event) {
    event.preventDefault();
    event.stopPropagation();
    const liItem = event.currentTarget;
    const submenu = liItem.querySelector('.submenu');
    const chevron = liItem.querySelector('.chevron-icon');
    if (submenu.style.display === 'block') {
      submenu.style.display = 'none';
      chevron.classList.remove('rotate');
    } else {
      submenu.style.display = 'block';
      chevron.classList.add('rotate');
    }
  }

  // Highlight menu when clicking
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function (event) {
      document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
      link.classList.add('active');
      if (link.closest('.submenu')) {
        const parentLink = link.closest('.submenu').parentElement.querySelector('.nav-link');
        if (parentLink) {
          parentLink.classList.add('active');
        }
      }
    });
  });

  // Generator data simulation
  document.addEventListener('DOMContentLoaded', function(){
    const startTimeStamp = Date.now();
    // Operate time
    function updateDuration(){
      const elapsed = Math.floor((Date.now() - startTimeStamp) / 1000);
      document.getElementById('duration').textContent = elapsed + ' sec';
    }
    setInterval(updateDuration, 1000);

    // Random data
    function getRandomInRange(min, max) {
      return (Math.random() * (max - min) + min).toFixed(2);
    }

    function updateDetailedData(){
      const voltageVal = getRandomInRange(219.6, 220.4);
      const currentVal = getRandomInRange(49.9, 50.1);
      const speedVal = getRandomInRange(2950, 3050);
      const freqVal = (speedVal / 60).toFixed(2);
      const powerVal = Math.round(voltageVal * currentVal);

      document.getElementById('voltage').textContent = voltageVal + ' V';
      document.getElementById('current').textContent = currentVal + ' A';
      document.getElementById('speed').textContent = Math.floor(speedVal) + ' rpm';
      document.getElementById('frequency').textContent = freqVal + ' Hz';
      document.getElementById('power').textContent = powerVal + ' W';

      const tempVal = getRandomInRange(60, 80);
      const oilVal = getRandomInRange(8, 10);
      const oxyVal = getRandomInRange(40, 50);
      const h2sVal = getRandomInRange(1, 2);

      document.getElementById('waterTemp').textContent = tempVal + ' °C';
      document.getElementById('oilPressure').textContent = oilVal + ' bar';
      document.getElementById('oxygen').textContent = oxyVal + ' %';
      document.getElementById('envH2s').textContent = h2sVal + ' %';
    }
    setInterval(updateDetailedData, 1000);

    // Voltage Chart format
    const voltageChart = new Chart(document.getElementById('voltageChart'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Phase A',
            data: [],
            borderColor: 'aqua',
            backgroundColor: 'aqua',
            pointStyle: 'circle',
            pointRadius: 0,
            pointHoverRadius: 0
          },
          {
            label: 'Phase B',
            data: [],
            borderColor: 'red',
            backgroundColor: 'red',
            pointStyle: 'circle',
            pointRadius: 0,
            pointHoverRadius: 0
          },
          {
            label: 'Phase C',
            data: [],
            borderColor: 'green',
            backgroundColor: 'green',
            pointStyle: 'circle',
            pointRadius: 0,
            pointHoverRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            labels:{
              usePointStyle: true
            },
            position: 'top' 
          },
          tooltip: { enabled: true }
        },
        scales: {
          x: { title: { display: false} },
          y: {
            title: { display: true, text: 'Voltage L-N (Average)' },
            min: 200,
            max: 240 
          }
        }
      }
    });

    // Current Chart format
    const currentChart = new Chart(document.getElementById('currentChart'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Phase A',
            data: [],
            borderColor: 'aqua',
            backgroundColor: 'aqua',
            pointStyle: 'circle',
            pointRadius: 0,
            pointHoverRadius: 0
          },
          {
            label: 'Phase B',
            data: [],
            borderColor: 'red',
            backgroundColor: 'red',
            pointStyle: 'circle',
            pointRadius: 0,
            pointHoverRadius: 0
          },
          {
            label: 'Phase C',
            data: [],
            borderColor: 'green',
            backgroundColor: 'green',
            pointStyle: 'circle',
            pointRadius: 0,
            pointHoverRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            labels: {
              usePointStyle: true
            },
            position: 'top'
          },
          tooltip: { enabled: true }
        },
        scales: {
          x: { title: { display: false} },
          y: {
            title: { display: true, text: 'Current L-N (Average)' },
            min: 45,
            max: 55 
          }
        }
      }
    });

    function updateCharts(){
      const now = new Date().toLocaleTimeString();

      const v1 = getRandomInRange(219.6, 220.4);
      const v2 = getRandomInRange(219.6, 220.4);
      const v3 = getRandomInRange(219.6, 220.4);

      const c1 = getRandomInRange(49.9, 50.1);
      const c2 = getRandomInRange(49.9, 50.1);
      const c3 = getRandomInRange(49.9, 50.1);

      if(voltageChart.data.labels.length >= 10){
        voltageChart.data.labels.shift();
        voltageChart.data.datasets.forEach(ds => ds.data.shift());
      }
      // Voltage Chart update data
      voltageChart.data.labels.push(now);
      voltageChart.data.datasets[0].data.push(v1);
      voltageChart.data.datasets[1].data.push(v2);
      voltageChart.data.datasets[2].data.push(v3);

      voltageChart.update();

      if(currentChart.data.labels.length >= 10){
        currentChart.data.labels.shift();
        currentChart.data.datasets.forEach(ds => ds.data.shift());
      }
      // Current Chart update data
      currentChart.data.labels.push(now);
      currentChart.data.datasets[0].data.push(c1);
      currentChart.data.datasets[1].data.push(c2);
      currentChart.data.datasets[2].data.push(c3);

      currentChart.update();
    }
    setInterval(updateCharts, 1000);
  });

  // Biogas Tank data simulation
  document.addEventListener('DOMContentLoaded', function () {

    const humidityEl = document.getElementById('humidityValue');
    const temperatureEl = document.getElementById('temperatureValue');
    const pressureEl = document.getElementById('pressureValue');
    const ch4El = document.getElementById('ch4Value');
    const h2sEl = document.getElementById('h2sValue');

    // Random data function in range
    function getRandomInRange(min, max) {
      return (Math.random() * (max - min) + min).toFixed(2);
    }

    // Update parameters value
    function updateSensor() {
      currentHumidity = Number(getRandomInRange(10, 30));
      currentTemperature = Number(getRandomInRange(60, 80));
      currentPressure = Number(getRandomInRange(10, 11));
      currentCH4 = Number(getRandomInRange(38, 44));
      currentH2S = Number(getRandomInRange(1, 10));

      humidityEl.textContent = `${currentHumidity} %`;
      temperatureEl.textContent = `${currentTemperature} °C`;
      pressureEl.textContent = `${currentPressure} bar`;
      ch4El.textContent = `${currentCH4} %`;
      h2sEl.textContent = `${currentH2S} ppm`;
    }

    setInterval(updateSensor, 1000);

    // Biogas Tank parameters chart
    const chartData = {
      labels: [], 
      datasets: [
        {
          label: 'Humidity (%)',
          data: [],
          borderColor: 'blue',
          backgroundColor: 'blue',
          pointStyle: 'circle',
          pointRadius: 0,
          pointHoverRadius: 0
        },
        {
          label: 'Temperature (°C)',
          data: [],
          borderColor: 'red',
          backgroundColor: 'red',
          pointStyle: 'circle',
          pointRadius: 0,
          pointHoverRadius: 0
        },
        {
          label: 'Pressure (bar)',
          data: [],
          borderColor: 'purple',
          backgroundColor: 'purple',
          pointStyle: 'circle',
          pointRadius: 0,
          pointHoverRadius: 0
        },
        {
          label: 'CH4 (ppm)',
          data: [],
          borderColor: 'aqua',
          backgroundColor: 'aqua',
          pointStyle: 'circle',
          pointRadius: 0,
          pointHoverRadius: 0
        },
        {
          label: 'H2S (ppm)',
          data: [],
          borderColor: 'black',
          backgroundColor: 'black',
          pointStyle: 'circle',
          pointRadius: 0,
          pointHoverRadius: 0
        }
      ]
    };

    // Chart config
    const chartConfig = {
      type: 'line',
      data: chartData,
      options: {
        responsive: true,
        plugins: {
          legend: {
            labels:{
              usePointStyle: true
            },
            position: 'top',
            // Legend mặc định cho phép ẩn/bật dataset khi nhấp vào
            onClick: (e, legendItem, legend) => {
              const index = legendItem.datasetIndex;
              const ci = legend.chart;
              ci.data.datasets[index].hidden = !ci.data.datasets[index].hidden;
              ci.update();
            }
          },
          tooltip: { enabled: true }
        },
        scales: {
          x: { 
            beginAtZero: true,
            title: { display: false}
          },
          y: {
            beginAtZero: true,
            max: 100, // Nếu cần cố định trục tung từ 0-100 (chỉ phù hợp với một số thông số)
            title: { display: false}
          }
        }
      }
    };

    const parameterChart = new Chart(
      document.getElementById('parameterChart'),
      chartConfig
    );

    // Real time update for chart
    function updateChart() {
      if (chartData.labels.length >= 30) {
        chartData.labels.shift();
        chartData.datasets.forEach(ds => ds.data.shift());
      }
 
      const now = new Date().toLocaleTimeString();
      chartData.labels.push(now);

      chartData.datasets[0].data.push(currentHumidity);
      chartData.datasets[1].data.push(currentTemperature);
      chartData.datasets[2].data.push(currentPressure);
      chartData.datasets[3].data.push(currentCH4);
      chartData.datasets[4].data.push(currentH2S);
      parameterChart.update();
    }

    setInterval(updateChart, 1000);
  });

  // Data Analysis 
  // Parameters of Biogas Tank
  const paramTank = [
    { value: 'humidity', label: 'Humidity (%)' },
    { value: 'temperature', label: 'Temperature (°C)' },
    { value: 'pressure', label: 'Pressure (bar)' },
    { value: 'ch4', label: 'CH4 (%)' },
    { value: 'h2s', label: 'H2S (ppm)' }
  ];

  //Parameters of Generators
  const paramGen = [
    { value: 'voltage', label: 'Voltage (V)' },
    { value: 'current', label: 'Current (A)' },
    { value: 'power', label: 'Power (W)' },
    { value: 'speed', label: 'Speed (rpm)' },
    { value: 'frequency', label: 'Frequency (Hz)' },
    { value: 'envTemperature', label: 'Temperature (°C)' },
    { value: 'oilPressure', label: 'Oil Pressure (bar)' },
    { value: 'oxygen', label: 'Oxygen (%)' },
    { value: 'genH2s', label: 'H2S (%)' },
    { value: 'runtime', label: 'Operate Time (hr)' }
  ];

  //
  function populateParameterSelect(device) {
    const parameterSelect = document.getElementById('parameterSelect');
    parameterSelect.innerHTML = '';

    let paramSet = [];
    if (device === 'biogasTank') {
      paramSet = paramTank;
    } else {
      paramSet = paramGen;
    }

    paramSet.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.value;
      opt.textContent = item.label;
      parameterSelect.appendChild(opt);
    });
  }


  document.getElementById('deviceSelect').addEventListener('change', function() {
    populateParameterSelect(this.value);
  });

  populateParameterSelect(document.getElementById('deviceSelect').value);

  function randomInRange(min, max) {
    return (Math.random() * (max - min) + min).toFixed(2);
  }

  document.getElementById('analysisForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const device = document.getElementById('deviceSelect').value;
    const parameter = document.getElementById('parameterSelect').value;
    const timeRange = document.getElementById('timeRangeSelect').value;
    const startDate = document.getElementById('startDate').value;
    
    // Choosing sample time based on time range
    let intervalMinutes;
    switch(timeRange) {
      case '1day':
        intervalMinutes = 10; // mỗi 10 phút
        break;
      case '1week':
        intervalMinutes = 60; // mỗi 1 giờ
        break;
      case '1month':
        intervalMinutes = 360; // mỗi 6 giờ
        break;
      case '6months':
        intervalMinutes = 1440; // mỗi 1 ngày
        break;
      default:
        intervalMinutes = 10;
    }

    let totalMinutes;
    switch(timeRange) {
      case '1day':
        totalMinutes = 1440;
        break;
      case '1week':
        totalMinutes = 7 * 1440;
        break;
      case '1month':
        totalMinutes = 30 * 1440;
        break;
      case '6months':
        totalMinutes = 6 * 30 * 1440;
        break;
      default:
        totalMinutes = 1440;
    }

    // Sample calculate
    const sampleCount = Math.floor(totalMinutes / intervalMinutes);

    let labels = [];
    let data = [];
    let start = new Date(startDate);

    for (let i = 0; i < sampleCount; i++) {
      let sampleTime = new Date(start.getTime() + i * intervalMinutes * 60000);
      labels.push(sampleTime.toLocaleString());

      // Data simulation
      if (device === 'biogasTank') {
        // Biogas Tank
        switch(parameter) {
          case 'humidity':
            data.push(randomInRange(10, 30));
            break;
          case 'temperature':
            data.push(randomInRange(60, 80));
            break;
          case 'pressure':
            data.push(randomInRange(10, 11));
            break;
          case 'ch4':
            data.push(randomInRange(38, 44));
            break;
          case 'h2s':
            data.push(randomInRange(1, 10));
            break;
          default:
            data.push(randomInRange(0, 100));
        }
      } else {
        // Generator
        switch(parameter) {
          case 'voltage':
            data.push(randomInRange(375, 385));
            break;
          case 'current':
            data.push(randomInRange(75, 85));
            break;
          case 'power':
            const v = randomInRange(375, 385);
            const c = randomInRange(75, 85);
            data.push((v * c).toFixed(0));
            break;
          case 'speed':
            data.push(randomInRange(2950, 3050));
            break;
          case 'frequency':
            const sp = randomInRange(2950, 3050);
            data.push((sp / 60).toFixed(2));
            break;
          case 'envTemperature':
            data.push(randomInRange(60, 80));
            break;
          case 'oilPressure':
            data.push(randomInRange(8, 10));
            break;
          case 'oxygen':
            data.push(randomInRange(40, 50));
            break;
          case 'genH2s':
            data.push(randomInRange(1, 2));
            break;
          case 'runtime':
            data.push((i * intervalMinutes / 60).toFixed(2));
            break;
          default:
            data.push(randomInRange(0, 100));
        }
      }
    }

    if (window.analysisChartInstance) {
      window.analysisChartInstance.destroy();
    }

    // Unit Label
    let unitLabel = '';
    if (device === 'biogasTank') {
      switch(parameter) {
        case 'humidity': unitLabel = '(%)'; break;
        case 'temperature': unitLabel = '(°C)'; break;
        case 'pressure': unitLabel = '(bar)'; break;
        case 'ch4': unitLabel = '(%)'; break;
        case 'h2s': unitLabel = '(ppm)'; break;
        default: unitLabel = '';
      }
    } else {
      // Generator
      switch(parameter) {
        case 'voltage': unitLabel = '(V)'; break;
        case 'current': unitLabel = '(A)'; break;
        case 'power': unitLabel = '(W)'; break;
        case 'speed': unitLabel = '(rpm)'; break;
        case 'frequency': unitLabel = '(Hz)'; break;
        case 'envTemperature': unitLabel = '(°C)'; break;
        case 'oilPressure': unitLabel = '(bar)'; break;
        case 'oxygen': unitLabel = '(%)'; break;
        case 'genH2s': unitLabel = '(%)'; break;
        case 'runtime': unitLabel = '(hr)'; break;
        default: unitLabel = '';
      }
    }

    const ctx = document.getElementById('analysisChart').getContext('2d');
    window.analysisChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: parameter + ' ' + unitLabel,
          data: data,
          borderColor: 'aqua',
          backgroundColor: 'aqua)',
          fill: false,
          pointRadius: 0,
          pointRadiusHover: 0,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: true }
        },
        scales: {
          x: {
            title: { display: false}
          },
          y: {
            title: { display: false}
          }
        }
      }
    });
  });

  function applyThresholdSettings() {
    // Gen ID
    const genID = document.getElementById('genIDSelect').value;
    
    // Biogas tank thresholds
    const biogasThreshold = {
      humidity: document.getElementById('thresholdHumidity').value,
      temperature: document.getElementById('thresholdTemperature').value,
      pressure: document.getElementById('thresholdPressure').value,
      ch4: document.getElementById('thresholdCH4').value,
      h2s: document.getElementById('thresholdH2S').value
    };
    
    // Generator thresholds
    const generatorThreshold = {
      genID: genID,
      voltage: document.getElementById('thresholdVoltage').value,
      current: document.getElementById('thresholdCurrent').value,
      speed: document.getElementById('thresholdSpeed').value,
      temp: document.getElementById('thresholdTempGen').value,
      oil: document.getElementById('thresholdOilGen').value,
      h2sGen: document.getElementById('thresholdH2SGen').value
    };
  }

  // Alarm examples
  const alarms = [
    {
      time: '2025-04-06 11:45:00',
      device: 'Biogas Tank',
      description: 'H2S concentration exceeded threshold',
    },
    {
      time: '2025-04-06 12:10:00',
      device: 'Generator G01',
      description: 'Current exceeded threshold',
    },
    {
      time: '2025-04-06 12:30:00',
      device: 'Generator G02',
      description: 'Voltage dropped down',
    },
    {
      time: '2025-04-06 13:45:00',
      device: 'Biogas Tank',
      description: 'Pressure exceeded threshold',
    },
  ];

  // Display alarms into table
  function displayAlarms(data) {
    const tbody = document.querySelector('#alarmTable tbody');
    tbody.innerHTML = '';
    data.forEach(alarm => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${alarm.time}</td>
        <td>${alarm.device}</td>
        <td>${alarm.description}</td>
      `;
      tr.addEventListener('click', () => showAlarmDetail(alarm));
      tbody.appendChild(tr);
    });
  }

  // Filters: Device and Time range
  function filterAlarms() {
    const deviceFilter = document.getElementById('deviceFilter').value;
    const startTimeValue = document.getElementById('startTimeFilter').value;
    const endTimeValue = document.getElementById('endTimeFilter').value;

    const filtered = alarms.filter(alarm => {
      if (deviceFilter && alarm.device !== deviceFilter) {
        return false;
      }
      const alarmDate = new Date(alarm.time);
      if (startTimeValue) {
        const startDate = new Date(startTimeValue);
        if (alarmDate < startDate) {
          return false;
        }
      }
      if (endTimeValue) {
        const endDate = new Date(endTimeValue);
        endDate.setHours(23,59,59,999); // bao gồm cả ngày End
        if (alarmDate > endDate) {
          return false;
        }
      }
      return true;
    });
    displayAlarms(filtered);
  } 
  displayAlarms(alarms);