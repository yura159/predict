$(document).on('click', '#enter-data', function(){
	var data = $('#text');
	var url = 'api/text';
	var dfr =$.ajax({
			type: 'POST',
			url: url,
			data: {
				'text': data.val(),
			},
			dataType: 'json'
		});
	dfr.done(function(answer){
		console.log('enter data');
		data.val('');
		$('#result').text(answer['text'] + ' answer');
	});
});