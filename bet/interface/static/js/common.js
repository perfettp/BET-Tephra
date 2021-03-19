
function page_tab_bind() {
    // for bootstrap 3 use 'shown.bs.tab', for bootstrap 2 use 'shown' in the next line
    $('#page_tabs').on("shown.bs.tab", 'a[data-toggle="tab"]', function (e) {
        var id = $(e.target).attr("href").substr(1);
        // window.location.hash = id;
        // save the latest tab; use cookies if you like 'em better:
        localStorage.setItem('lastTab', $(this).attr('href'));
    })

    var hash = window.location.hash;
    if (hash) {
        $('[href="' + hash + '"]').tab('show');
        localStorage.setItem('lastTab', hash);
    } else {
        // go to the latest tab, if it exists:
        var lastTab = localStorage.getItem('lastTab');
        if (lastTab) {
            $('[href="' + lastTab + '"]').tab('show');
        }
    }
}

function offset_tab_bind(tabs_id) {
    // for bootstrap 3 use 'shown.bs.tab', for bootstrap 2 use 'shown' in the next line
    $(tabs_id).on("shown.bs.tab", 'a[data-toggle="tab"]', function (e) {
        var t_id = '#' + $(e.target).attr("href").substr(1);
        // window.location.hash = id;
        // save the latest tab; use cookies if you like 'em better:
        localStorage.setItem('lastOffset', $(this).attr('href'));

        update_tab_maps(t_id);
    })

    var first_inner_li = $(tabs_id + ' li[class="enabled"]:first')
    var first_inner_a = $(tabs_id + ' li[class="enabled"]:first a[data-toggle="tab"]')
    var first_inner_tab = $(first_inner_a.attr('href'))

    first_inner_li.addClass('active')
    first_inner_tab.addClass('in active')

    update_tab_maps('#'+first_inner_tab.attr("id"));

}

function update_tab_maps (tab_id) {
        var tab_maps = []
        $( tab_id + ' div[class="map-canvas"]').each(
            function() { tab_maps.push($(this).attr("id")) })

        $(tab_maps).each(function(i) {
            setTimeout(function(){
                ol_maps[tab_maps[i]].updateSize();
                }, 0);
        });
}