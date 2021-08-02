$(document).ready(function() {

    $('.image-card-btn').on('click', function() {
        var data_id = $(this).attr('data_id')
        var page = $(this).attr('page')
        var label = $(this).attr('label')

        req = $.ajax({
            url: '/update-card',
            type: 'POST',
            data: {data_id: data_id, page: page, label: label}
        });

        req.done(function (data){
            if(label==='Unsure'){
                $('#id'+data_id).removeClass('unsure').addClass('correct');
                $('#id-a-'+data_id).attr('label', 'Correct');
            }else if(label==='Incorrect'){
                $('#id'+data_id).removeClass('incorrect').addClass('unsure');
                $('#id-a-'+data_id).attr('label', 'Unsure');
            }else if(label==='Correct'){
                $('#id' + data_id).removeClass('correct').addClass('incorrect');
                $('#id-a-' + data_id).attr('label', 'Incorrect');
            }
        });
    });

    $('.image-card-btn').bind("contextmenu",function(e){
        e.preventDefault();
        var data_id = $(this).attr('data_id')
        var page = $(this).attr('page')
        var label = $(this).attr('label')
        var strSlice = 0;

        req = $.ajax({
            url: '/get_slice',
            type: 'POST',
            data: {data_id: data_id, page: page, slice: strSlice}
        });

        req.done(function (data) {
            console.log(data);

            $('#rangeSlices').attr("min", "-"+data.halfLen);
            $('#rangeSlices').attr("max", data.halfLen);
            $('#rangeSlices').attr("value", strSlice);
            $('#rangeSlices').attr("data_id", data_id);
            $('#rangeSlices').attr("page", page);

            $('#minSlice').html("-"+data.halfLen);
            $('#maxSlice').html(data.halfLen);

            $('#cardDetails').addClass(label.toLowerCase());
            $('#imgDetails').attr("src", "data:image/jpeg;base64," + data.data);
            $('#detailsModal').modal("show");
        });
    });

    $('#rangeSlices').change( function() {
        var rangeValue = $(this).val();
        console.log(rangeValue)
        var data_id = $(this).attr('data_id')
        var page = $(this).attr('page')

        req = $.ajax({
            url: '/get_slice',
            type: 'POST',
            data: {data_id: data_id, page: page, slice: rangeValue}
        });

        req.done(function (data){
            $('#imgDetails').attr("src", "data:image/jpeg;base64," + data.data);
        });
    });

});