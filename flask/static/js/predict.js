$("#form-predict").submit(function (event) {
    event.preventDefault()

    inputGRE = $("#inputGRE").val()
    inputToefl = $("#inputToefl").val()
    inputUniRating = $("#inputUniRating").val()
    inputSOP = $("#inputSOP").val()
    inputLOR = $("#inputLOR").val()
    inputCGPA = $("#inputCGPA").val()
    inputResearch = $("#inputResearch").val()

    $.ajax({
        url: 'http://localhost:5000/predict',
        contentType: 'application/json',
        cache: false,
        method: 'POST',
        data: JSON.stringify({
            inputGRE: inputGRE,
            inputToefl: inputToefl,
            inputUniRating: inputUniRating,
            inputSOP: inputSOP,
            inputLOR: inputLOR,
            inputCGPA: inputCGPA,
            inputResearch: inputResearch
        }),
        dataType: 'json',
        success: function (data) {
            var result = parseFloat(data).toFixed(1);
            console.log('Response Successful!')

            if (result < 0) {
                Swal.fire({
                    icon: 'error',
                    title: 'Oops...',
                    text: 'Unfortunately, admission chance is very low!',
                    footer: 'Chance predicted: 0.0 %'
                })
            } else {
                if (result < 0.5) {
                    Swal.fire({
                        icon: 'error',
                        title: 'Oops...',
                        text: 'Unfortunately, admission chance very low!',
                        footer: 'Chance predicted: ' + result * 100 + '%'
                    })
                } else {
                    if (result < 1) {
                        Swal.fire(
                            'Well done!',
                            'You have a good chance of being admitted!<br/>Chance predicted: ' + result * 100 + '%',
                            'success'
                        )
                    } else {
                        Swal.fire(
                            'Well done!',
                            'You have a good chance of being admitted!<br/>Chance predicted: ' + result * 100 + '%',
                            'success'
                        )
                    }
                }
            }

            $('#form-predict')[0].reset();
        }
    })


});

// class="was-validated"