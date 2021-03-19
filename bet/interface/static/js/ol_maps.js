
var ol_maps = {};


function render_map(map_id, legend_id, kml_url){
//    console.debug("rendering map " + map_id + " " + legend_id)
    var vector = new ol.layer.Vector({
      source: new ol.source.Vector({
        format: new ol.format.KML(),
        url: kml_url
      })
    });

    center = ol.proj.fromLonLat([14.1821813, 40.8040395])
    var map = new ol.Map({
        layers: [
            new ol.layer.Tile({
              source: new ol.source.OSM()
            }),
            vector
        ],
        target: map_id,
        controls:  ol.control.defaults().extend([
            new ol.control.ScaleLine()
        ]),
        view: new ol.View({
            center: center,
            zoom: 9
        })
    });
    ol_maps[map_id] = map;

    var legend = document.getElementById(legend_id);

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
