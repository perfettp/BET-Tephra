// Return standard error with 95% confidence
function se95(p, n) {
    return Math.sqrt(p*(1-p)/n)*1.96;
};

var magma_width = $("#magma_chart").width();
//var magma_height = (magma_width/3) * 2;
var magma_height = 200;

// Settings
//var width = 440,
//    height = 400,
//    padding = 30;


var magma_margin = {
    'top': 30,
    'right': 35,
    'bottom': 30,
    'left': 30
};


magma_margin.hor = magma_margin.left + magma_margin.right;
magma_margin.ver = magma_margin.top + magma_margin.bottom;

var width = magma_width - (magma_margin.hor) ,
    height = magma_height,
    padding = 30;

// Config
var dataset = "/static/data/magma-demo.csv",
    parseDate = d3.time.format("%Y/%m").parse,
    electionDate = "",  // "2014/11"
    interpolation = "linear";

var coalitionLeft = ["A", "B", "F", "Ø"],
    coalitionLeftColor = "#D7191C", // blue
    coalitionRight = ["V", "O", "K", "I", "C"],
    coalitionRightColor = "#2B83BA", // red
    displaySingleCoalition = false;
    // false, "left", "right"

var useCoalitionLabels = true,
    magma_yAxisTitle = "Votes (%)",
    cutoff = 50;

if (useCoalitionLabels === true) { magma_margin.right = 50; }
// for the labels; 40 + 10 for each array.length > 4

var magma_x = d3.time.scale()
    .range([0, width]);

var magma_y = d3.scale.linear()
    .range([height, 0]);

var magma_xAxis = d3.svg.axis()
    .scale(magma_x)
    .orient("bottom")
    .ticks(7)
    .tickSubdivide(2);

var magma_yAxis = d3.svg.axis()
    .scale(magma_y)
    .orient("left")
    .tickFormat(d3.format(".0%"));

var lineLeft = d3.svg.area()
    .interpolate(interpolation)
    .x(function(d) { return magma_x(d["date"]); })
    .y(function(d) { return magma_y(d["left"]); });

var lineRight = d3.svg.area()
    .interpolate(interpolation)
    .x(function(d) { return magma_x(d["date"]); })
    .y(function(d) { return magma_y(d["right"]); });

var confidenceAreaLeft = d3.svg.area()
    .interpolate(interpolation)
    .x(function(d) { return magma_x(d["date"]); })
    .y0(function(d) {
        return magma_y(d["left"] - d["confidenceRight"]); })
    .y1(function(d) {
        return magma_y(d["left"] + d["confidenceRight"]); });

var confidenceAreaRight = d3.svg.area()
    .interpolate(interpolation)
    .x(function(d) { return magma_x(d["date"]); })
    .y0(function(d) {
        return magma_y(d["right"] - d["confidenceRight"]); })
    .y1(function(d) {
        return magma_y(d["right"] + d["confidenceRight"]); });

var magma_svg = d3.select("#magma_chart").append("svg")
    .attr({
        "width": width + magma_margin.left + magma_margin.right,
        "height": height + magma_margin.top + magma_margin.bottom
    })
    .append("g")
    .attr("transform", "translate(" + magma_margin.left + "," + magma_margin.top + ")");

d3.csv(dataset, function(error, data) {
    data.forEach(function(d) {
        d.date = parseDate(d.date);
        d.respondents = parseFloat(d.respondents);

        coalitionSum = function(d, coalition) {
            var votes = parseFloat(0);
            for (var i = 0; i < coalition.length; i++) {
                votes += parseFloat(d[coalition[i]])
            }
            // Return percentage in decimal format
            return votes>1 ? votes/100 : votes;
        };
        d["left"] = coalitionSum(d, coalitionLeft),
        d["right"] = coalitionSum(d, coalitionRight),
        d["total"] = d["left"] + d["right"],
        d["confidenceLeft"] = se95(d["left"], d["respondents"]),
        d["confidenceRight"] = se95(d["right"], d["respondents"]);
    });

    if (electionDate === "") {
        magma_x.domain(d3.extent(data, function(d) {
            return d.date; }));
    } else {
        magma_x.domain([
            d3.min(data, function(d) { return d.date; }),
            parseDate(electionDate)
        ]);
    }
    magma_y.domain([
        d3.min(data, function(d) {
            var min = Math.min(d["right"], d["left"]);
            return min - se95(min, d["respondents"]);
        }),
        d3.max(data, function(d) {
            var max = Math.max(d["right"], d["left"]);
            return max + se95(max, d["respondents"]);
        })
    ]);

    magma_svg.datum(data);

    // X axis
    magma_svg.append("g")
        .attr({
            "class": "x axis",
            "transform": "translate(0," + height + ")"
        })
        .call(magma_xAxis);

    // Y axis
    magma_svg.append("g")
        .attr("class", "y axis")
        .call(magma_yAxis)
        .append("text")
        .attr({
            "transform": "rotate(-90)",
            "y": 6,
            "dy": ".71em"
        })
        .style("text-anchor", "end")
        .text(magma_yAxisTitle);

    // Confidence area
    if (displaySingleCoalition !== "right") {
        magma_svg.append("path")
            .attr({
                "class": "area confidence",
                "fill": coalitionLeftColor,
                "d": confidenceAreaLeft
            });
    }
    if (displaySingleCoalition !== "left") {
        magma_svg.append("path")
            .attr({
                "class": "area confidence",
                "fill": coalitionRightColor,
                "d": confidenceAreaRight
            });
    }

    // Lines
    if (displaySingleCoalition !== "right") {
        magma_svg.append("path")
            .attr({
                "class": "line",
                "d": lineLeft,
                "stroke": coalitionLeftColor
            });
    }
    if (displaySingleCoalition !== "left") {
        magma_svg.append("path")
            .attr({
                "class": "line",
                "d": lineRight,
                "stroke": coalitionRightColor
            });
    }

    // Dots
    var dots = magma_svg.selectAll("circle")
        .data(data)
        .enter();

    if (displaySingleCoalition !== "right") {
        var dotsLeft = dots.append("circle")
            .attr({
                "class": "dot",
                "r": 3,
                "cx": lineLeft.x(),
                "cy": lineLeft.y(),
                "stroke": coalitionLeftColor
            });
    }
    if (displaySingleCoalition !== "left") {
        var dotsRight = dots.append("circle")
            .attr({
                "class": "dot",
                "r": 3,
                "cx": lineRight.x(),
                "cy": lineRight.y(),
                "stroke": coalitionRightColor
            });
    }

    // Divider
    magma_svg.append("line")
        .attr("class", "divider")
        .attr({
            "x1": magma_x.range()[0],
            "x2": magma_x.range()[1],
            "y1": magma_y(cutoff),
            "y2": magma_y(cutoff)
        });

    // Graph label
    if (useCoalitionLabels === true) {
        if (displaySingleCoalition !== "right") {
            magma_svg.append("text")
                .data(data)
                .attr("transform", function(d) {
                    return "translate(" + magma_x(data[data
                    .length-1]["date"]) + "," + magma_y(data[data.length-1]["left"]) + ")"; })
                .attr({
                    "x": 10,
                    "dy": ".35em",
                    "class": "label",
                    "id": "coalitionLeft"
                })
                .text(coalitionLeft.join(""));
        }
        if (displaySingleCoalition !== "left") {
            magma_svg.append("text")
                .data(data)
                .attr("transform", function(d) {
                    return "translate(" + magma_x(data[data
                    .length-1]["date"]) + "," + magma_y(data[data.length-1]["right"]) + ")"; })
                .attr({
                    "x": 10,
                    "dy": ".35em",
                    "class": "label",
                    "id": "coalitionRight"
                })
                .text(coalitionRight.join(""));
        }
    }
});
