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
                $('#id'+data_id).removeClass('bg-light').addClass('bg-success');
                $('#id-a-'+data_id).attr('label', 'Correct');
            }else if(label==='Incorrect'){
                $('#id'+data_id).removeClass('bg-danger').addClass('bg-light');
                $('#id-a-'+data_id).attr('label', 'Unsure');
            }else if(label==='Correct'){
                $('#id' + data_id).removeClass('bg-success').addClass('bg-danger');
                $('#id-a-' + data_id).attr('label', 'Incorrect');
            }
        });
    });
});