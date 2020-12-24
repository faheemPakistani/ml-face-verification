window.onload = function(){
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


}