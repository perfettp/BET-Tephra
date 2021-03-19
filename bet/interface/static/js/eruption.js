function addAxesAndLegend (eruption_svg, eruption_xAxis, eruption_yAxis, margin, chartWidth, chartHeight) {
  var legendWidth  = 200,
      legendHeight = 100;

  // clipping to make sure nothing appears behind legend
  eruption_svg.append('clipPath')
    .attr('id', 'axes-clip')
    .append('polygon')
      .attr('points', (-margin.left)                 + ',' + (-margin.top)                 + ' ' +
                      (chartWidth - legendWidth - 1) + ',' + (-margin.top)                 + ' ' +
                      (chartWidth - legendWidth - 1) + ',' + legendHeight                  + ' ' +
                      (chartWidth + margin.right)    + ',' + legendHeight                  + ' ' +
                      (chartWidth + margin.right)    + ',' + (chartHeight + margin.bottom) + ' ' +
                      (-margin.left)                 + ',' + (chartHeight + margin.bottom));

  var eruption_axes = eruption_svg.append('g')
    .attr('clip-path', 'url(#axes-clip)');

  eruption_axes.append('g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,' + chartHeight + ')')
    .call(eruption_xAxis);

  eruption_axes.append('g')
    .attr('class', 'y axis')
    .call(eruption_yAxis)
    .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 6)
      .attr('dy', '.71em')
      .style('text-anchor', 'end')
      .text('Time (s)');

//  var legend = eruption_svg.append('g')
//    .attr('class', 'legend')
//    .attr('transform', 'translate(' + (chartWidth - legendWidth) + ', 0)');
//
//  legend.append('rect')
//    .attr('class', 'legend-bg')
//    .attr('width',  legendWidth)
//    .attr('height', legendHeight);
//
//  legend.append('rect')
//    .attr('class', 'outer')
//    .attr('width',  75)
//    .attr('height', 20)
//    .attr('x', 10)
//    .attr('y', 10);
//
//  legend.append('text')
//    .attr('x', 115)
//    .attr('y', 25)
//    .text('5% - 95%');
//
//  legend.append('rect')
//    .attr('class', 'inner')
//    .attr('width',  75)
//    .attr('height', 20)
//    .attr('x', 10)
//    .attr('y', 40);
//
//  legend.append('text')
//    .attr('x', 115)
//    .attr('y', 55)
//    .text('25% - 75%');
//
//  legend.append('path')
//    .attr('class', 'median-line')
//    .attr('d', 'M10,80L85,80');
//
//  legend.append('text')
//    .attr('x', 115)
//    .attr('y', 85)
//    .text('Median');
}

function drawPaths (eruption_svg, data, x, y) {
  var upperOuterArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct95); })
    .y1(function (d) { return y(d.pct75); });

  var upperInnerArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct75); })
    .y1(function (d) { return y(d.pct50); });

  var medianLine = d3.svg.line()
    .interpolate('basis')
    .x(function (d) { return x(d.date); })
    .y(function (d) { return y(d.pct50); });

  var lowerInnerArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct50); })
    .y1(function (d) { return y(d.pct25); });

  var lowerOuterArea = d3.svg.area()
    .interpolate('basis')
    .x (function (d) { return x(d.date) || 1; })
    .y0(function (d) { return y(d.pct25); })
    .y1(function (d) { return y(d.pct05); });

  eruption_svg.datum(data);

  eruption_svg.append('path')
    .attr('class', 'area upper outer')
    .attr('d', upperOuterArea)
    .attr('clip-path', 'url(#rect-clip)');

  eruption_svg.append('path')
    .attr('class', 'area lower outer')
    .attr('d', lowerOuterArea)
    .attr('clip-path', 'url(#rect-clip)');

  eruption_svg.append('path')
    .attr('class', 'area upper inner')
    .attr('d', upperInnerArea)
    .attr('clip-path', 'url(#rect-clip)');

  eruption_svg.append('path')
    .attr('class', 'area lower inner')
    .attr('d', lowerInnerArea)
    .attr('clip-path', 'url(#rect-clip)');

  eruption_svg.append('path')
    .attr('class', 'median-line')
    .attr('d', medianLine)
    .attr('clip-path', 'url(#rect-clip)');
}

function makeChart (data) {
  var eruption_width = $("#eruption_chart").width();
  var eruption_height = 200;
  var svgWidth  = eruption_width,
      svgHeight = eruption_height,
      margin = { top: 20, right: 20, bottom: 40, left: 40 },
      chartWidth  = svgWidth  - margin.left - margin.right,
      chartHeight = svgHeight - margin.top  - margin.bottom;

  var eruption_x = d3.time.scale().range([0, chartWidth])
            .domain(d3.extent(data, function (d) { return d.date; })),
     eruption_y = d3.scale.linear().range([chartHeight, 0])
            .domain([0, d3.max(data, function (d) { return d.pct95; })]);

  var eruption_xAxis = d3.svg.axis().scale(eruption_x).orient('bottom')
                .innerTickSize(-chartHeight).outerTickSize(0).tickPadding(10),
      eruption_yAxis = d3.svg.axis().scale(eruption_y).orient('left')
                .innerTickSize(-chartWidth).outerTickSize(0).tickPadding(10);

  var eruption_svg = d3.select('#eruption_chart').append('svg')
    .attr('width',  svgWidth)
    .attr('height', svgHeight)
    .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  // clipping to start chart hidden and slide it in later
  var rectClip = eruption_svg.append('clipPath')
    .attr('id', 'rect-clip')
    .append('rect')
      .attr('width', chartWidth)
      .attr('height', chartHeight);

  addAxesAndLegend(eruption_svg, eruption_xAxis, eruption_yAxis, margin, chartWidth, chartHeight);
  drawPaths(eruption_svg, data, eruption_x, eruption_y);

}

var eruption_parseDate  = d3.time.format('%Y-%m-%d').parse;
d3.json('/static/data/eruption-demo.json', function (error, rawData) {
  if (error) {
    console.error(error);
    return;
  }

  var eruption_data = rawData.map(function (d) {
    return {
      date:  eruption_parseDate(d.date),
      pct05: d.pct05 / 1000,
      pct25: d.pct25 / 1000,
      pct50: d.pct50 / 1000,
      pct75: d.pct75 / 1000,
      pct95: d.pct95 / 1000
    };
  });

  d3.json('/static/data/eruption-markers.json', function (error, markerData) {
    if (error) {
      console.error(error);
      return;
    }

    makeChart(eruption_data);
  });
});
