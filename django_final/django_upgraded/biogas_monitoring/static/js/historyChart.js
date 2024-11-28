$(document).ready(function(){
    $('#historySubmit').click(function(){
        dataRealTime1.datasets[0].label = $('#listSensorHistory option:selected').text();
        $.ajax({
            url: '/chart',
            type: 'POST',
            dataType: 'json',
            data: JSON.stringify({
                "idMachine": idMachine,
                "sensor": $('#listSensorHistory').val(),
                "start": new Date($("#startTime").val()).getTime(),
                "end": new Date($("#endTime").val()).getTime(),
            }),
            contentType:"application/json; charset=UTF-8"
        }).done(function(data) {
            labels1.length = 0;
            dataRealTime1.datasets[0].data.length = 0;
            Object.keys(data).forEach(function(key) {
                //console.log(key, data[key]);
                time = (new Date(Number(key))).toLocaleDateString()+" "+(new Date(Number(key))).toLocaleTimeString();
                labels1.push(time);
                dataRealTime1.datasets[0].data.push(data[key]);

            });
            //console.log(labels1);
            //console.log(dataRealTime1.datasets[0].data);
            myChart1.update();
        })
    })
})