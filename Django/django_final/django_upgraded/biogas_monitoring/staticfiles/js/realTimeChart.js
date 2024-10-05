$(document).ready(function(){
    $('#realTimeSubmit').click(function(){
        typeSensor = $('#listSensorRealTime').val();
        dataRealTime2.datasets[0].label = $('#listSensorRealTime option:selected').text();
        labels2.length = 0;
        dataRealTime2.datasets[0].data.length = 0;
        myChart2.reset();
    })
})
