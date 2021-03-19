

function render_map(map_id, legend_id, kml_url){

    var exampleLoc = new google.maps.LatLng(40.8040395, 14.1821813);
    var mapProp = {
        center: exampleLoc,
        zoom:9,
        disableDefaultUI: true,
        zoomControl: true,
        zoomControlOptions: {
          style: google.maps.ZoomControlStyle.SMALL
        },
        mapTypeId: google.maps.MapTypeId.HYBRID
    };
    var map = new google.maps.Map(document.getElementById(map_id),
    mapProp);

    var exampleKml = new google.maps.KmlLayer({
      url: kml_url,
      clickable: false,
      preserveViewport:true
    });

    exampleKml.setMap(map);
    var legend = document.getElementById(legend_id);
    map.controls[google.maps.ControlPosition.RIGHT_TOP].push(legend);

    $.get(kml_url, function(data) {
        $(data).find("Placemark").each(function(index, value){
                name = $(this).find("name").text();
                style = $(this).find("styleUrl").text().replace("#","");
                color = $(data).find("Style[id='" + style + "']").find("LineStyle").find("color").text()


                var color_div = document.createElement('div');
                color_div.className = 'color-box'
                c = '#' + color.slice(6,8) + color.slice(4,6) + color.slice(2,4)
                color_div.style.backgroundColor= c

                color_div.innerHTML = name + ' kg/m^2'
                legend.appendChild(color_div);

                //output as a navigation

            })

    })

}


//<div class="input-color">
//    <input type="text" value="Orange" />
//    <div class="color-box" style="background-color: #FF850A;"></div>
//    <!-- Replace "#FF850A" to change the color -->
//</div>
