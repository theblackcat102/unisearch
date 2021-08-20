const node = document.getElementsByClassName("search-bar-action")[0];
node.addEventListener("keyup", function(event) {
    if (event.key === "Enter") {
        console.log("submit form");
        document.getElementsByClassName("search-form-submit")[0].submit();
    }
});



function submitImage() {
    document.getElementsByClassName("search-image")[0].submit();
}

function uploadAndSubmit() {
    document.getElementById("image-input").click();
};

function textImageSubmit() {
    const text_input = document.getElementsByClassName("search-bar-action")[0];
    const submit_input = document.getElementById("image-text-input");
    submit_input.value = text_input.value;
    document.getElementsByClassName("search-image-form")[0].submit();   
}