//declearing html elements
let a = 1;
const imgDivs = document.querySelector(".profile-pic");
const imgs = document.querySelector("#photos");
const files = document.querySelector("#files");
const uploadBtns = document.querySelector("#uploadBtns");

const imgDiv = document.querySelector(".profile-pic-div");
const img = document.querySelector("#photo");
const file = document.querySelector("#file");
const uploadBtn = document.querySelector("#uploadBtn");


uploadBtns.addEventListener('click', function(){
  a = 2;
})
uploadBtn.addEventListener('click', function(){
  a = 1;
})

imgDivs.addEventListener("mouseenter", function () {
  uploadBtns.style.display = "block";
});

imgDivs.addEventListener("mouseleave", function () {
  uploadBtns.style.display = "none";
});

imgDiv.addEventListener("mouseenter", function () {
  uploadBtn.style.display = "block";
});

imgDiv.addEventListener("mouseleave", function () {
  uploadBtn.style.display = "none";
});



file.addEventListener("change", function () {
    const choosedFile = this.files[0];

    if (choosedFile) {
      const read = new FileReader(); //FileReader is a predefined function of JS
      if(a > 1){
      read.addEventListener("load", function () {
        imgs.setAttribute("src", read.result);
      });
    }
    else{
      console.log(img.getAttribute('src'))
      read.addEventListener("load", function () {
        img.setAttribute("src", read.result);
      });
    }
      read.readAsDataURL(choosedFile);
    }
  });



