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
});