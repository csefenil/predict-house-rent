  // Your web app's Firebase configuration
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  var firebaseConfig = {
    apiKey: "AIzaSyBv76-kMxF4sPnX0mFJxfvFu1f0MgqxJDU",
    authDomain: "myapplication-280d8.firebaseapp.com",
    databaseURL: "https://myapplication-280d8-default-rtdb.firebaseio.com",
    projectId: "myapplication-280d8",
    storageBucket: "myapplication-280d8.appspot.com",
    messagingSenderId: "1053891401810",
    appId: "1:1053891401810:web:6864aeced75051da2ee78b",
    measurementId: "G-NTNPQ57E2V"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  firebase.analytics();

/*
$("#sqftbutton").click(function(){
    var sqft = $("persqft").val();
});
*/

function RetrieveRegion(){
    firebase.database().ref("Region").once("value",
    function(record){
        var sqft = CurrentRecord.val().persqft;
        AddItem(persqft);
    });
}
window.onload = RetrieveRegion;

function AddItem(){
    var tbody = document.getElementById('tbody1');
    var trow = document.createElement('tr');
    var td1 = document.createElement('td');
    td1.innerHTML =sqft;
    trow.appendChild(td1);
    tbody.appendChild(trow);
}