$(document).ready(function () {
	//TODO: ENTER the correct url.
    console.log("1");
    var rootURL="http://127.0.0.1:5000/result";
    var pos;
    var domainName ;
    var id ;
    var current_fs, next_fs, previous_fs; //fieldsets
    var left, opacity, scale; //fieldset properties which we will animate
    var animating; //flag to prevent quick multi-click glitches
    $(".loading").hide();
    $("#fs2").hide();
    $("#submit1").click(function(e){
    	e.preventDefault();
    	sex = $("#sex").val();
    	var age = $("[name=age]").val();
    	pneumonia= $("#pneumonia").val();
    	diabetes= $("#diabetes").val();
    	copd= $("#copd").val();
    	asthama= $("#asthma").val();
    	inmsupr= $("#inmsupr").val();
    	cardiovascular= $("#cardiovascular").val();
    	obesity= $("#obesity").val();
    	renal_chronic= $("#renal_chronic").val();
    	tobacco= $("#tobacco").val();
    	
    	var jsonObj = {"sex":sex, "age" : age, "pneumonia" :pneumonia, "diabetes":diabetes,
    	 "copd":copd, "asthma":asthama, "inmsupr":inmsupr, "cardiovascular":cardiovascular,"obesity":obesity,
    	"renal_chronic":renal_chronic,"tobacco":tobacco}
    	
    	console.log(JSON.stringify(jsonObj));
    
    	$.ajax({
    		type :'GET',
    		url  : "/result",
    		contentType : "application/json",
    		data : jsonObj,
    		crossDomain : true,
    		success : function (result){
    			$("#submit1").hide();
    			prediction_result = JSON.parse(result); 
    			icu = prediction_result.icu;
    			hospital = prediction_result.hos;
    
    			if(animating) return false;
    			animating = true;
    
    			current_fs = $(this).parent();
    			next_fs = $(this).parent().next();
    
    			//activate next step on progressbar using the index of next_fs
    			$("#progressbar li").eq($("fieldset").index(next_fs)).addClass("active");
    
    			//show the next fieldset
    			$("#fs1").hide();
    			$("#fs2").show();
    			//hide the current fieldset with style
    			
    			document.getElementById("details").innerHTML = "ICU Required : "+icu+"<br>"+"Hospitalization Required: "+hospital ;
    		},
    		error: function(xhr,textstatus,error)
    		{
    			var eval = JSON.parse(xhr.responseText);
    			alert("Error :"+"\n"+eval.code+"\n"+eval.status+"\n"+eval.description);
    		}
    	});
    });
});
