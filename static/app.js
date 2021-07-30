$(document).ready(function() {

    $('.image-card-btn').on('click', function() {
        var data_id = $(this).attr('data_id')
        var page = $(this).attr('page')
        var label = $(this).attr('label')

        req = $.ajax({
            url: '/update-card',
            type: 'GET',
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
        var slices_before = $(this).attr('slices_b')
        var slices_after = $(this).attr('slices_a')

        req = $.ajax({
            url: '//get_slice_before',
            type: 'GET',
            data: {data_id: data_id, page: page}
        });

        req.done(function (data) {
            console.log(data)
            $('#cardDetails').addClass(label.toLowerCase());
            $('#imgDetails').attr("src", '{{"data:image/jpeg;base64,"+' + data[0] + "}}");
            $('#detailsModal').modal("show");
        });
    });
});