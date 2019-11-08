$("#form-predict").submit(function (event) {

    // NOTE Aqui devo fazer uma requisição de pref GET para o servidor Flask
    // que retornará a resposta contendo o predict value
    alert("Submissão efetuada!");
    event.preventDefault();
});

// class="was-validated"