$(document).ready(function () {

    function label(lbl) {
        return $.trim(lbl.replace(/[ .]/g, '-').replace('+-', '').toLowerCase());
    }

    // a hack: macos doesn't support cuda, so disable all cuda options when it
    // is selected.
    function disableCuda() {
        $('.install .option').each(function(){
            if (label($(this).text()).indexOf("cuda") != -1) {
                $(this).addClass('disabled');
            }
        });
    }
    function enableCuda() {
        $('.install .option').each(function(){
            if (label($(this).text()).indexOf("cuda") != -1) {
                $(this).removeClass('disabled');
            }
        });
    }

    // find the user os, and set the according option to active
    function setActiveOSButton() {
        var os = "linux"
        var agent = window.navigator.userAgent.toLowerCase();
        if (agent.indexOf("win") != -1) {
            os = "windows"
        } else if (agent.indexOf("mac") != -1) {
            os = "macos"
        }
        if (os == "macos") {
            disableCuda();
        }
        $('.install .option').each(function(){
            if (label($(this).text()).indexOf(os) != -1) {
                $(this).addClass('active');
            }
        });
    }

    setActiveOSButton();

    // apply theme
    function setTheme() {
        $('.opt-group .option').each(function(){
            $(this).addClass('mdl-button mdl-js-button mdl-js-ripple-effect mdl-button--raised ');
            $(this).attr('id', label($(this).text()));
        });
        $('.opt-group .active').each(function(){
            $(this).addClass('mdl-button--colored');
        });
    }
    setTheme();


    // show the command according to the active options
    function showCommand() {
        $('.opt-group .option').each(function(){
            $('.'+label($(this).text())).hide();
            // console.log('disable '+label($(this).text()));
        });
        $('.opt-group .active').each(function(){
            $('.'+label($(this).text())).show();
            // console.log('enable '+label($(this).text()));
        });
    }
    showCommand();

    function setOptions() {
        var el = $(this);
        el.siblings().removeClass('active');
        el.siblings().removeClass('mdl-button--colored');
        el.addClass('active');
        el.addClass('mdl-button--colored');
        // console.log('enable'+el.text())
        // console.log('disable'+el.siblings().text())
        console.log($('.install #macos').hasClass('active') )
        if ($('.install #macos').hasClass('active') == true) {
            disableCuda();
        } else {
            enableCuda();
        }
        showCommand();
    }

    $('.opt-group').on('click', '.option', setOptions);

});
