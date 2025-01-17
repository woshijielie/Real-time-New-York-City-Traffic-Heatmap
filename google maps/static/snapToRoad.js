// https://gist.github.com/derzorngottes/3b57edc1f996dddcab25
var apiKey = config.gcp_api_key;

var tag = "#";
var map;
var drawingManager;
var placeIdArray = [];
var polylines = [];
var clearNeedLines = [];
var snappedCoordinates = [];
var lineIndex = [];
var histTime;
var weatherList = ["Clear", "Clouds", "Tornado", "Rain", "Fog"];
var color = ["#00FF00", "#FF8C00", "#FF0000"];
var start_time = "2022_05_05_09_00";
var minutesToRound = 15;

function loadMap() {
    var script = document.createElement("script");
    script.src = "https://maps.googleapis.com/maps/api/js?libraries=drawing,places&key=" + apiKey;
    document.getElementsByTagName("head")[0].appendChild(script);
}

function playbackButtonControl(controlDiv) {
    const outer = document.createElement("div");
    const playButton = document.createElement("div");

    outer.style.ppadding = "1.5em";
    outer.style.width = "2.6em";
    outer.style.marginLeft = "10px";
    outer.style.marginTop = "22px";
    outer.style.backgroundColor = "rgba(0, 0, 0, .25)";
    outer.style.cursor = "pointer";
    outer.style.marginBottom = "100px";

    playButton.style.margin = "0 auto";
    playButton.style.top = "25%";
    playButton.style.position = "relative";
    playButton.style.width = "0";
    playButton.style.height = "0"
    playButton.style.borderWidth = "1.3em 0 1.3em 2.6em";
    playButton.style.borderColor = "transparent transparent transparent #000";
    playButton.style.borderStyle = "solid";
    playButton.style.opacity = ".75";
    playButton.id = "play";

    outer.appendChild(playButton);
    controlDiv.appendChild(outer);

    playButton.addEventListener("click", function (event) {
        event.preventDefault();
        playButton.style.borderStyle = "double";
        playButton.style.borderWidth = "0 0 0 2.6em";
        setInterval(playBackTraffic, 6000);
    });

    return playButton;
}

function weatherButtonControl(controlDiv) {
    const controlUI = document.createElement("select");

    controlUI.style.backgroundColor = "#fff";
    controlUI.style.border = "2px solid #fff";
    controlUI.style.borderRadius = "3px";
    controlUI.style.boxShadow = "0 2px 6px rgba(0,0,0,.3)";
    controlUI.style.cursor = "pointer";
    controlUI.style.marginTop = "22px";
    controlUI.style.fontSize = "20px";
    controlUI.style.width = "80px";
    // controlUI.style.width = "80px";
    controlUI.value = "1";
    controlUI.onchange = "convey_weather()";
    controlUI.id = "weatherSelect";

    for (var i = 1; i <= weatherList.length; i++) {
        const optionUI = document.createElement("option")
        optionUI.value = i.toString();
        optionUI.append(weatherList[i - 1]);
        controlUI.append(optionUI);
    }

    controlDiv.append(controlUI);

    return controlUI;
}

function dateButtonControl(controlDiv) {
    // Set CSS for the control border.
    const controlUI = document.createElement("input");

    controlUI.style.backgroundColor = "#fff";
    controlUI.style.border = "2px solid #fff";
    controlUI.style.borderRadius = "3px";
    controlUI.style.boxShadow = "0 2px 6px rgba(0,0,0,.3)";
    controlUI.style.cursor = "pointer";
    controlUI.style.marginBottom = "22px";
    controlUI.style.fontSize = "16px";
    // controlUI.style.textAlign = "center";
    controlUI.type = "datetime-local";
    controlUI.name = "local-time";
    controlUI.id = "history";
    controlUI.title = "Select the history";
    controlUI.pattern = "yyyy-MM-dd HH:mm";
    // controlUI.onchange =  "convey_time()";
    controlDiv.appendChild(controlUI);

    return controlUI;
}

function initialize() {
    var mapOptions = {
        zoom: 12,
        center: {lat: 40.7724, lng: -73.9841,}
    };
    map = new google.maps.Map(document.getElementById('map'), mapOptions);

    const weatherControlDiv = document.createElement("div");
    const weatherButtonUI = weatherButtonControl(weatherControlDiv);
    map.controls[google.maps.ControlPosition.TOP_CENTER].push(weatherControlDiv);

    const dateControlDiv = document.createElement("div");
    const dateTimeUI = dateButtonControl(dateControlDiv);
    map.controls[google.maps.ControlPosition.BOTTOM_CENTER].push(dateControlDiv);

    const playControlDiv = document.createElement("div");
    const playButtonUI = playbackButtonControl(playControlDiv);
    map.controls[google.maps.ControlPosition.TOP_CENTER].push(playControlDiv);
    // Adds a Places search box. Searching for a place will center the map on that
    // location.
    map.controls[google.maps.ControlPosition.RIGHT_TOP].push(
        document.getElementById('bar'));
    var autocomplete = new google.maps.places.Autocomplete(
        document.getElementById('autoc'));
    autocomplete.bindTo('bounds', map);
    autocomplete.addListener('place_changed', function () {
        var place = autocomplete.getPlace();
        if (place.geometry.viewport) {
            map.fitBounds(place.geometry.viewport);
        } else {
            map.setCenter(place.geometry.location);
            map.setZoom(17);
        }
    });

    // Weather now
    weatherButtonUI.value = weather;

    // Marker
    for (var i = 0; i < points.length; i++) {
        const location = {lat: points[i][0][0], lng: points[i][0][1]};
        const info = points[i][1];
        if (info[6] === '3') {
            Marker_and_infoWin(location, map, info);
        }
    }

    var interval_id = setInterval(getTime, 1000);
    dateTimeUI.addEventListener('click', function (event) {
        event.preventDefault();
        clearInterval(interval_id);
        $("#history").on("click", function () {
            convey_time();
        })
        // clearInterval(interval_id);
    });

    weatherButtonUI.addEventListener('click', function (event) {
        event.preventDefault();
        clearInterval(interval_id);
        $("#weatherSelect").change(function () {
            weather = weatherButtonUI.value;
            convey_weather();
        });
    })

    playButtonUI.addEventListener('click', function (event) {
        event.preventDefault();
        clearInterval(interval_id);
    });

    runSnapToRoad();

    // Clear button. Click to remove all polylines.
    document.getElementById('clear').addEventListener('click', function (event) {
        event.preventDefault();
        for (var i = 0; i < polylines.length; ++i) {
            polylines[i].setMap(null);
        }
        polylines = [];
        return false;
    });
}

function clearRoad() {
    for (var i = 0; i < clearNeedLines.length; ++i) {
        clearNeedLines[i].setMap(null);
    }
    clearNeedLines = [];
    placeIdArray = [];

}

// Snap a user-created polyline to roads and draw the snapped path
function runSnapToRoad() {
    clearNeedLines = polylines;
    polylines = [];
    var limit = 100; // The upper limit of the number of points we need to draw each time
    for (var i = 0; i < len; i++) {
        var step = -1;
        var index = 0;
        pathValues = [];
        var rLen = result[i].length;
        var times = parseInt(rLen / limit);
        while (index <= times) {
            lineIndex.push(rating[i]);
            if (rLen < (index + 1) * limit) {
                pathValues = result[i].slice(index * limit, rLen);
            }
            pathValues = result[i].slice(index * limit, (index + 1) * limit);
            // strToLatlng(pathValues);
            // console.log(pathValues);
            // pathValues = [new google.maps.LatLng(40.754782543, -73.97993647238523).toUrlValue(), new google.maps.LatLng(40.75733682890686, -73.98599451733426).toUrlValue(), new google.maps.LatLng(40.75670781705963, -73.986450492866920).toUrlValue(), new google.maps.LatLng(40.756526801242345, -73.9865886266312).toUrlValue(), new google.maps.LatLng(40.75627734019398, -73.9867777223668).toUrlValue(), new google.maps.LatLng(40.75605333874007, -73.98694267822125).toUrlValue()];
            index++;

            $.get('https://roads.googleapis.com/v1/snapToRoads', {
                interpolate: true,
                key: apiKey,
                path: pathValues.join('|'),
            }, function (data) {
                step++;
                processSnapToRoadResponse(data);
                drawSnappedPolyline(step);
            });
        }
    }
}

function strToLatlng(pathValues) {
    var len = pathValues.length;
    var g_pathValues = []
    for (var i = 0; i < len; i++) {
        var coor = pathValues[i].split(",");
        var lat = parseFloat(coor[0]);
        var lng = parseFloat(coor[1]);
        g_pathValues.push(new google.maps.LatLng(lat, lng).toUrlValue());
    }

    pathValues = g_pathValues;
}

// Store snapped polyline returned by the snap-to-road service.
function processSnapToRoadResponse(data) {
    snappedCoordinates = [];
    placeIdArray = [];
    for (var i = 0; i < data.snappedPoints.length; i++) {
        var latlng = new google.maps.LatLng(
            data.snappedPoints[i].location.latitude,
            data.snappedPoints[i].location.longitude);
        snappedCoordinates.push(latlng);
        placeIdArray.push(data.snappedPoints[i].placeId);
    }
}

// function sleep(delay) {
//   var start = (new Date()).getTime();
//   while ((new Date()).getTime() - start < delay) {
//     continue;
//   }
// }

// Draws the snapped polyline (after processing snap-to-road response).
function drawSnappedPolyline(index) {
    // var gradient = new gradientColor("#00FF00","#FF0000", 251);

    var step = 0;
    if (lineIndex[index] >= -10) {
        step = 0;
    } else if (lineIndex[index] < -10 && lineIndex[index] >= -100) {
        step = 1;
    } else {
        step = 2;
    }

    var snappedPolyline = new google.maps.Polyline({
        path: snappedCoordinates,
        strokeColor: color[step],
        // strokeColor: gradient[step],
        strokeWeight: 4,
        strokeOpacity: 0.8,
    });

    snappedPolyline.setMap(map);

    polylines.push(snappedPolyline);
}

function Marker_and_infoWin(location, map, info) {
    const win_model =
        '<div id="content">' +
        '<div id="siteNotice">' +
        "</div>" +
        '<h1 id="firstHeading" class="firstHeading">Bad Event</h1>' +
        '<div id="bodyContent">' + info +
        "</div>" +
        "</div>";

    const marker = new google.maps.Marker({
        position: location,
        title: "Bad Event",
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 6,
        },
        draggable: false,
        map: map,
    });

    const infowindow = new google.maps.InfoWindow({
        content: win_model,
    });

    marker.addListener("click", () => {
        infowindow.open({
            anchor: marker,
            map,
            shouldFocus: false,
        });
    });
}

function convey_weather() {
    $.ajax({
        type: "POST",
        url: "/get_history_weather",
        dataType: "json",
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(weather),
        success: function (response) {
            console.log("success");
            var dateControl = document.querySelector("#history");
            if (response == "None") {
                window.alert("No such weather in last 24h");
                clearRoad();
                dateControl.value = "";
            } else {
                result = response[0];
                rating = response[1];
                weather = response[2];
                len = response[3];
                history_time = response[4]

                dateControl.value = history_time;

                runSnapToRoad();
                clearRoad();
            }
        },
        error: function (request, status, error) {
            console.log("Error");
            console.log(request)
            console.log(status)
            console.log(error)
        }
    });
}

function convey_time() {
    var dateControl = document.querySelector("#history");
    var history = dateControl.value;
    var min = history.substr(history.length - 2);
    min = Math.round(min / minutesToRound) * minutesToRound;
    if (min == 60) {
        min == "00"
    }
    histTime = history;
    $.ajax({
        type: "POST",
        url: "/get_history_time",
        dataType: "json",
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(histTime),
        success: function (response) {
            console.log("success");
            console.log(response);
            result = response[0];
            rating = response[1];
            weather = response[2];
            len = response[3];

            var weatherControl = document.querySelector("#weatherSelect");
            weatherControl.value = weather;
            clearRoad();
            runSnapToRoad();
        },
        error: function (request, status, error) {
            console.log("Error");
            console.log(request)
            console.log(status)
            console.log(error)
        }
    });
}

function playBackTraffic() {
    $.ajax({
        type: "POST",
        url: "/playBack",
        dataType: "json",
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(start_time),
        success: function (response) {
            console.log("success");
            if (response == "Finish") {
                return
            }
            // var obj  = eval( '('+ response + ')' );
            result = response[0];
            rating = response[1];
            weather = response[2];
            len = response[3];
            start_time = response[4];
            console.log(start_time);
            var dateControl = document.querySelector("#history");
            dateControl.value = start_time;

            var weatherControl = document.querySelector("#weatherSelect");
            weatherControl.value = weather;
            clearRoad();
            runSnapToRoad();
        },
        error: function (request, status, error) {
            console.log("Error");
            console.log(request)
            console.log(status)
            console.log(error)
        }
    });
}

function getTime() {
    var today = new Date();
    var yyyy = today.getFullYear();
    var MM = today.getMonth() + 1;
    var dd = today.getDate();
    var hh = today.getHours();
    var mm = today.getMinutes();
    mm = Math.round(mm / minutesToRound) * minutesToRound
    if (mm == 60) {
        mm = 0;
        hh += 1;
    }
    MM = checkTime(MM);
    dd = checkTime(dd);
    hh = checkTime(hh);
    mm = checkTime(mm);

    var time = yyyy + "-" + MM + "-" + dd + "T" + hh + ":" + mm;
    $("#history").attr("max", time);

    var dateControl = document.querySelector("#history");
    dateControl.value = time;

    function checkTime(i) {
        if (i < 10) {
            i = "0" + i;
        }
        return i;
    }
}

// gradient color setting
class gradientColor {
    constructor(startColor, endColor, step) {
        var startRGB = this.colorRgb(startColor); // Convert to rgb array mode
        var startR = startRGB[0];
        var startG = startRGB[1];
        var startB = startRGB[2];

        var endRGB = this.colorRgb(endColor);
        var endR = endRGB[0];
        var endG = endRGB[1];
        var endB = endRGB[2];

        var sR = (endR - startR) / step; // Final difference
        var sG = (endG - startG) / step;
        var sB = (endB - startB) / step;

        var colorArr = [];
        for (var i = 0; i < step; i++) {
            // Calculate the hex value of each step
            var hex = this.colorHex('rgb(' + parseInt((sR * i + startR)) + ',' + parseInt((sG * i + startG)) + ',' + parseInt((sB * i + startB)) + ')');
            colorArr.push(hex);
        }
        return colorArr;
    }

    colorRgb(sColor) {
        var reg = /^#([0-9a-fA-f]{3}|[0-9a-fA-f]{6})$/;
        var sColor = sColor.toLowerCase();
        if (sColor && reg.test(sColor)) {
            if (sColor.length === 4) {
                var sColorNew = "#";
                for (var i = 1; i < 4; i += 1) {
                    sColorNew += sColor.slice(i, i + 1).concat(sColor.slice(i, i + 1));
                }
                sColor = sColorNew;
            }
            var sColorChange = [];
            for (var i = 1; i < 7; i += 2) {
                sColorChange.push(parseInt("0x" + sColor.slice(i, i + 2)));
            }
            return sColorChange;
        } else {
            return sColor;
        }
    }

    colorHex(rgb) {
        var _this = rgb;
        var reg = /^#([0-9a-fA-f]{3}|[0-9a-fA-f]{6})$/;
        if (/^(rgb|RGB)/.test(_this)) {
            var aColor = _this.replace(/(?:(|)|rgb|RGB)*/g, "").split(",");
            var strHex = "#";
            for (var i = 0; i < aColor.length; i++) {
                var hex = Number(aColor[i]).toString(16);
                hex = hex < 10 ? 0 + '' + hex : hex; // 保证每个rgb的值为2位
                if (hex === "0") {
                    hex += hex;
                }
                strHex += hex;
            }
            if (strHex.length !== 7) {
                strHex = _this;
            }
            return strHex;
        } else if (reg.test(_this)) {
            var aNum = _this.replace(/#/, "").split("");
            if (aNum.length === 6) {
                return _this;
            } else if (aNum.length === 3) {
                var numHex = "#";
                for (var i = 0; i < aNum.length; i += 1) {
                    numHex += (aNum[i] + aNum[i]);
                }
                return numHex;
            }
        } else {
            return _this;
        }
    }
}

window.onload = loadMap();
$(window).load(initialize);
