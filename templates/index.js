
//  db.ref('region').once('value',   function(snapshot) {
//    snapshot.forEach(function(childSnapshot) {
//      var childKey = childSnapshot.key;
//      var childData = childSnapshot.val();
//      console.log({childKey , childData})
//      // ...
//    });
//  });

// var regions = db.collection('region').get().then((snapshot) => {
//     snapshot.forEach(doc => {
//         console.log( "xyz", doc.data() , doc);
//     })
// console.log(snapshot)
// }).catch((error) => {
//         console.log("Error getting documents: ", error);
//     });




let btn = document.getElementById("getpersqfeet")
let locality = document.getElementById("locality")

let reciveValue = document.getElementById("reciveValue");

btn.addEventListener("click" , ()=>{
    console.log("click button" , locality.value , data.Region)
    if(data.Region.hasOwnProperty(locality.value))
    {
        reciveValue.value = data.Region[locality.value].persqft
    }
    else{
        alert("value not found in database")
    }
})


/*
$("#sqftbutton").click(function(){
    var sqft = $("persqft").val();
});
*/
/*

function RetrieveRegion(){
    firebase.database().ref("Region").once("value",
    function(record){
        var sqft = CurrentRecord.val().persqft;
        AddItem(persqft);
    });
}
window.onload = RetrieveRegion;
*/
