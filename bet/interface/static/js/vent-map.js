var centerLonLat = [14.1821813, 40.8040395];
var centerWebMercator = ol.proj.fromLonLat(centerLonLat)

// Debug object
// var geojsonObject =  {
//                        "type": "FeatureCollection",
//                        "features": [
//                            {"geometry": {"coordinates": centerWebMercator,
//                                          "type": "Point"},
//                             "properties": {"val": 5.0}, "type": "Feature"}
//                        ],
//                     }


var image = new ol.style.Circle({
                    radius: 5,
                    fill: new ol.style.Fill({
                                        color: 'rgba(255,0,0,0.2)'
                            }),
                    stroke: new ol.style.Stroke({color: 'black', width: 1})
            });

var styles = {
                'Point': [new ol.style.Style({
                            image: image
                })],
                'Circle': [new ol.style.Style({
                            stroke: new ol.style.Stroke({
                                        color: 'red',
                                        width: 2
                                    }),
                            fill: new ol.style.Fill({
                                        color: 'rgba(255,0,0,0.2)'
                            })
                           })]
             }

var styleFunction = function(feature, resolution) {
                        if (feature.getGeometry().getType()=='Point') {

                            return [new ol.style.Style({
                                image:  new ol.style.Circle({
                                              radius: 5,
                                              fill: new ol.style.Fill({
                                                color: '#'+Math.floor(Math.random()*16777215).toString(16)
                                              }),
                                              stroke: new ol.style.Stroke({
                                                 color: 'black',
                                                 width: 1})
                            })})]
                        } else {
                            return styles[feature.getGeometry().getType()];
                        }
                    };

function init() {

    var osm_source =  new ol.source.OSM()

    var geoportale_source = new ol.source.TileWMS({
               url: "http://wms.pcn.minambiente.it/ogc?map=/ms_ogc/WMS_v1.3/Vettoriali/Carta_geologica.map",
               params:  {layer: 'GE.CARTA_GEOLOGICA'}
             })
    var demo_source = new ol.source.TileWMS({
     url: 'http://demo.opengeo.org/geoserver/wms',
     params: {LAYERS: 'ne:NE1_HR_LC_SR_W_DR', VERSION: '1.1.1'}
   })

    var tile_layer = new ol.layer.Tile({
            source: osm_source
        })

    // render the map
    var map = new ol.Map({
        target: 'vent_map',
        renderer: 'canvas',
        layers: [tile_layer],
        view: new ol.View({
            center: centerWebMercator,
            zoom: 12
        })
    });


    d3.json("static/bet_ef_data.geojson", function (error, data_points) {


        // Reproject data. This could be done better.
        for (i_point in data_points.features) {

            data_points.features[i_point].geometry.coordinates =
                ol.proj.fromLonLat(data_points.features[i_point].geometry.coordinates)
        }

        data_projection = (new ol.format.GeoJSON()).readProjection(data_points)
        // console.debug(data_projection)
        features = (new ol.format.GeoJSON()).readFeatures(data_points)

        // vector layer
        var vector_source = new ol.source.Vector()

        var vector_layer = new ol.layer.Vector({
                source: vector_source,
                style: styleFunction
        //        style: new ol.style.Style({
        //                    stroke: new ol.style.Stroke({
        //                        color: '#00dd00',
        //                        width: 4
        //                    }),
        //                    fill: new ol.style.Fill({
        //                        color: '#ffdd00'
        //                    }),
        //        })
            });

        vector_source.addFeatures(features)
        var numFeatures = vector_layer.getSource().getFeatures().length;
        console.log("Count right after construction: " + numFeatures);
        // vector_source.addFeature(new ol.Feature(new ol.geom.Point(centerWebMercator)));
        map.addLayer(vector_layer)
    });
}

init()


