function updateGuiRealTimeValue(name, rpi, value, unit){
    //let doc_element = document.getElementById(rpi+"Ia");
    let doc_element = document.getElementById(rpi+name);
    //If it isn't "undefined" and it isn't "null", then it exists.
    if(typeof(doc_element) != 'undefined' && doc_element != null){
        document.getElementById(rpi+name).innerHTML = value + unit;
    } 
}

// function add data electrical to html
function getDataElectrical(data, rpi) {
    data.data.forEach(element => {
        let temp_id = element.id;
        temp_id = temp_id.replace(rpi,'');
        switch (temp_id) {
            case "eles":
                updateGuiRealTimeValue("eles", rpi, element.v, " vòng/phút");
                break;
            case "eleva":
                updateGuiRealTimeValue("eleva", rpi, element.v, " V");
                break;
            case "elevb":
                updateGuiRealTimeValue("elevb", rpi, element.v, " V");
                break;
            case "elevc":
                updateGuiRealTimeValue("elevc", rpi, element.v, " V");
                break;
            case "elevna":
                updateGuiRealTimeValue("elevna", rpi, element.v, " V");
                break;
            case "elevab":
                updateGuiRealTimeValue("elevab", rpi, element.v, " V");
                break;
            case "elevbc":
                updateGuiRealTimeValue("elevbc", rpi, element.v, " V");
                break;
            case "elevca":
                updateGuiRealTimeValue("elevca", rpi, element.v, " V");
                break;
            case "elevla":
                updateGuiRealTimeValue("elevla", rpi, element.v, " V");
                break;
            case "eleia":
                updateGuiRealTimeValue("eleia", rpi, element.v, " A");
                break;
            case "eleib":
                updateGuiRealTimeValue("eleib", rpi, element.v, " A");
                break;
            case "eleic":
                updateGuiRealTimeValue("eleic", rpi, element.v, " A");
                break;
            case "eleiav":
                updateGuiRealTimeValue("eleiav", rpi, element.v, " A");
                break;
            case "elepwa":
                data = (element.v)/1000;
                data = data.toFixed(2);
                updateGuiRealTimeValue("elepwa", rpi, data, " kW");
                break;
            case "elepwb":
                data = (element.v)/1000;
                data = data.toFixed(2);
                updateGuiRealTimeValue("elepwb", rpi, data, " kW");
                break;
            case "elepwc":
                data = (element.v)/1000;
                data = data.toFixed(2);
                updateGuiRealTimeValue("elepwc", rpi, data, " kW");
                break;
            case "elepwt":
                data = (element.v)/1000;
                data = data.toFixed(2);
                updateGuiRealTimeValue("elepwt", rpi, data, " kW");
                break;
            case "elepfa":
                updateGuiRealTimeValue("elepfa", rpi, element.v, " ");
                break;
            case "elepfb":
                updateGuiRealTimeValue("elepfb", rpi, element.v, " ");
                break;
            case "elepfc":
                updateGuiRealTimeValue("elepfc", rpi, element.v, " ");
                break;
            case "elepft":
                updateGuiRealTimeValue("elepft", rpi, element.v, " ");
                break;
            case "elef":
                updateGuiRealTimeValue("elef", rpi, element.v, " Hz");
                break;
            case "eleewh":
                data = Math.floor((element.v)*100);  
                data /= 100;         
                updateGuiRealTimeValue("eleewh", rpi, data, " kWh");
                break;
            case "eleevah":
                data = Math.floor((element.v)*100);
                data /= 100;  
                updateGuiRealTimeValue("eleevah", rpi, data, " kVAh");
                break;
            case "eletop":
                data = Math.floor((element.v)*100);
                data /= 100;  
                updateGuiRealTimeValue("eletop", rpi, data, " h");
                break;
            case "elethdva":
                updateGuiRealTimeValue("elethdva", rpi, element.v, " %");
                break;
            case "elethdvb":
                updateGuiRealTimeValue("elethdvb", rpi, element.v, " %");
                break;
            case "elethdvc":
                updateGuiRealTimeValue("elethdvc", rpi, element.v, " %");
                break;
            case "elethdia":
                updateGuiRealTimeValue("elethdia", rpi, element.v, " %");
                break;
            case "elethdib":
                updateGuiRealTimeValue("elethdib", rpi, element.v, " %");
                break;
            case "elethdic":
                updateGuiRealTimeValue("elethdic", rpi, element.v, " %");
                break;

        }

    });
}

// function add data environment to html
function getDataEvironment(data, rpi) {
    data.data.forEach(element => {
        let temp_id = element.id;
        temp_id = temp_id.replace(rpi,'');
        switch (temp_id) {
            case "envtw":
                updateGuiRealTimeValue("envtw", rpi, element.v, " ℃");
                break;
            case "envpo":
                updateGuiRealTimeValue("envpo", rpi, element.v, " Bar");
                break;
            case "envo2":
                updateGuiRealTimeValue("envo2", rpi, element.v, " lambpda");
                break;
            case "envh2s":
                if (element.v < 0) {
                    updateGuiRealTimeValue("envh2s", rpi, 50, " ppm");
                    break; 
                } else {
                    updateGuiRealTimeValue("envh2s", rpi, element.v, " ppm");
                    break;
                }

        }
    });
}

// function add data environment to html
function getDataOperation(data, rpi) {
    data.data.forEach(element => {
        let temp_id = element.id;
        temp_id = temp_id.replace(rpi,'');
        switch (temp_id) {
            case "opepidsp":
                updateGuiRealTimeValue("opepidsp", rpi, element.v, " rpm");
                break;
            case "opepidout":
                updateGuiRealTimeValue("opepidout", rpi, element.v, " %");
                break;
            case "opevpb":
                updateGuiRealTimeValue("opevpb", rpi, element.v, " ");
                break;  
            case "opepb":
                updateGuiRealTimeValue("opepb", rpi, element.v, " mm");
                break;  
            case "opevsfb":
                updateGuiRealTimeValue("opevsfb", rpi, element.v, " %");
                break;                
            case "opetb":
                let dateTimeStamp = new Date().getTime();
                let timeBeginTimeStamp = Number(element.v)
                let timeOperate = (dateTimeStamp - timeBeginTimeStamp)/1000>>0;
                let timeOperateH = timeOperate/3600>>0;
                let timeOperateM = (timeOperate%3600)/60>>0;
                let timeOperateS = (timeOperate%3600)%60;

                let operateTime = String(timeOperateH) +":"+ String(timeOperateM) +":"+ String(timeOperateS);
                let timeBegin = new Date(timeBeginTimeStamp);
                let timeBeginTime = timeBegin.toLocaleTimeString();
                let timeBeginDate = timeBegin.toLocaleDateString();
                //console.log(timeBeginDate);
                updateGuiRealTimeValue("daytb", rpi, timeBeginDate, "");
                updateGuiRealTimeValue("timetb", rpi, timeBeginTime, "")
                updateGuiRealTimeValue("op", rpi, operateTime, "")
                break;
        }
    });
}

let typeSensor = "eles";
function getDataRealTimeChart(typeSensor, data, idMachine){
    try{
        let checkId = document.getElementById(idMachine+'realTimeChart');
        if(checkId != null){
            if (data.rpi == idMachine){
                sensor = data.rpi + typeSensor;
                data.data.forEach(function(element){
                    if (element.id == sensor){
                        time = (new Date(Number(data.time))).toLocaleTimeString();
                        //let dateTime = new Date((data.time)*1000);
                        //let time = dateTime.getHours()+":"+dateTime.getMinutes()+":"+dateTime.getSeconds()
                        labels2.push(time);
                        dataRealTime2.datasets[0].data.push(element.v);
                        myChart2.update();
                    };
                });
            }
        }
    }
    catch{
        console.log("error draw chart")
    }
}

// called when a message arrives
async function onMessageArrived(message) {
    if(message) {
        try {
            let data = JSON.parse(message.payloadString);
            //console.log(data);
            let topic = message.destinationName;
            if (topic == "sensor_data") {
                // revise this piece of code later

                let listRpi = ['g01','g02','g03','g04','g05','g06','g07','g08','g09','g10','g11','g12','g13','g14','g15','g16','g99'];
                if (listRpi.indexOf(data.rpi) != -1) {
                    await getDataRealTimeChart(typeSensor,data, idMachine);
                    if (data.type == "electrical") {
                        getDataElectrical(data, data.rpi);
                    } else if (data.type == "environment") {
                        getDataEvironment(data, data.rpi)
                    } else if (data.type == "operation") {
                        getDataOperation(data, data.rpi)
                    }
                }
            }
        } catch(e) {
            console.log("no data request"); // error in the above string (in this case, yes)!
        }
    }
}


