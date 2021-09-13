const program = require('commander');
const rp = require('request-promise');
var fs = require('fs');
var http = require('http');
var mkdirp = require('mkdirp');
var getDirName = require('path').dirname;

const checkResourceFormat = (format) => {
	//console.log(format.toLowerCase());

	if (format.toLowerCase() === "json"){
		return true;
	}
	if (format.toLowerCase() === "csv"){
		return true;
	}	


	return false;
}

const processDatasetResponses = (datasetResponses) => {
	console.log("processDatasetResponses datasetResponses.length " + datasetResponses.length);

	var responseObject = {}
	for(var i = 0; i < datasetResponses.length; i++) {
		var datasetResponse = datasetResponses[i];
		for (var j = 0; j < datasetResponse.body.result.results.length; j++){
			

			if (datasetResponse.body.result.results[j].name !== datasetResponse.datasetName) {
				// console.log(datasetResponse.body.result.results[j].name + " !== " + datasetResponse.datasetName);
				continue;
			}

			// console.log(datasetResponse.body.result.results[j].name + " === " + datasetResponse.datasetName);
			// console.log(datasetResponse.body.result.results[j].name + " " + datasetResponse.body.result.results[j].resources.length);

			var resourceObjList = []
			for (var k = 0; k < datasetResponse.body.result.results[j].resources.length; k++){
				var resource = datasetResponse.body.result.results[j].resources[k];

				if (!checkResourceFormat(resource.format)){
					continue;
				}

				var resourceObj = {
					name: resource.name.toLowerCase(),
					format: resource.format.toLowerCase(),
					url: resource.url,
				}

				//console.log(JSON.stringify(resourceObj));
				resourceObjList.push(resourceObj);
			}

			var tagObjList = []
			for (var k = 0; k < datasetResponse.body.result.results[j].tags.length; k++){
				var tag = datasetResponse.body.result.results[j].tags[k];

				var tagObj = {
					name: tag.name.toLowerCase(),
					display_name: tag.display_name.toLowerCase(),
				}
				tagObjList.push(tagObj);						
			}

			var groupObjList = []
			for (var k = 0; k < datasetResponse.body.result.results[j].groups.length; k++){
				var group = datasetResponse.body.result.results[j].groups[k];

				var groupObj = {
					name: group.name.toLowerCase(),
					display_name: group.display_name.toLowerCase(),
					description: group.description.toLowerCase(),
				}
				groupObjList.push(groupObj);							
			}

			var datasetObj = {
				title: datasetResponse.body.result.results[j].title.toLowerCase(),
				notes: datasetResponse.body.result.results[j].notes.toLowerCase(),
				count: datasetResponse.body.result.count,

				resources: resourceObjList,
				tags: tagObjList,
				groups: groupObjList,
			}

			responseObject[datasetResponse.datasetName] = datasetObj;
			//console.log(JSON.stringify(responseObject[datasetResponse.datasetName]));
		}
	}	

	//console.log(JSON.stringify(responseObject));
	return responseObject;
}

const requestPromise = (req) => {
  return rp({
    uri: req,
    json: true,
    resolveWithFullResponse: true
  });
}

const requestDataset = (req) => {
	//console.log("http://data.surrey.ca/api/3/action/package_search?q=" + req);
	return rp({
	    uri: "http://data.surrey.ca/api/3/action/package_search?q=" + req,
	    json: true,
	    resolveWithFullResponse: true
  	}).then( (response) => {
  		//console.log("http://data.surrey.ca/api/3/action/package_search?q=" + req);
  		response.datasetName = req;
  		console.log(response.datasetName);
  		return response;
  	}).catch(function(err){
			console.log("requestDataset error " + err); 
	});
}

const requestDataCatalogue = function() {
  requestPromise("http://data.surrey.ca/api/3/action/package_list")
  .then( (response) => {
    	console.log("response.length: " + response.body.result.length);
		// console.log("response.count: " + response.body.result.count);    	

		var promises = Promise.all(response.body.result.map(requestDataset));

		promises.then( (datasetResponses) => {
			console.log("datasetResponses.length: " + datasetResponses.length);
			return processDatasetResponses(datasetResponses);
		})
		.then( (responseObject) => {
			var responseStr = JSON.stringify(responseObject);
			//console.log("response: " + responseStr);
			fs.writeFile("./downloadResourceURL.json", responseStr, function(err) {
			    if(err) {
			        return console.log(err);
			    }
			});  			
			return responseStr;			
		}).catch(function(err){
			console.log("error " + err); 
		});

  });	
}

const surreyGetData = function() {
	return new Promise((resolve, reject) => {
		requestDataCatalogue((err, result) => {
		      if (err) {
		        return reject(err)
		      }
		      return resolve(result)
		});
	});
}

const downloadResource = function(filepath, url) {

	mkdirp(getDirName(filepath), function (err) {
		if (err) throw err;

		var file = fs.createWriteStream(filepath);
		console.log(filepath + " " + url);

		var request = http.get(url, function(response) {
			// if (request.statusCode != 200) {
			// 	console.log("error " + url);
			// }
		  	response.pipe(file);
		  	//console.log("success " + url + " " + filepath);
		})
		.on("error", function (){console.log("error " + url)}); 					
	});		
}

program
  .version('1')
  .description('Project: Schema Integration for Heterogenous Data Sources');

// node index.js downloadCatalogue
program
  .command('downloadCatalogue')
  .description('download the URLs of the datasets')
  .action(function(){
    //console.log('calling surreyGetData()');

    surreyGetData()
    
  })

// node index.js downloadDatasets
program
  .command('downloadDatasets')
  .description('download the datasets using URL')
  .action(function(){

	fs.readFile('./downloadResourceURL.json', 'utf8', function(err, data) {  
	    if (err) throw err;
	    var catalogue = JSON.parse(data);

		var catalogueKeys = [];
		for(var k in catalogue) catalogueKeys.push(k);
		//console.log(catalogueKeys.length + catalogueKeys)

	    for (var i = 0; i < catalogueKeys.length; i++) {
	    	var dataset = catalogue[catalogueKeys[i]];
	    	//console.log(dataset.title);

	    	for (var j = 0; j < dataset.resources.length; j++) {
	    		var resource = dataset.resources[j];

	    		//console.log(resource.url);
	    		var n = resource.url.lastIndexOf('/');
				var filename = resource.url.substring(n + 1);

				var filepath = "./" + catalogueKeys[i] + "/" + filename

				// var missingResources = ["http://cosmos.surrey.ca/geo_ref/Images/OpenDataArchives/morgan_heights_neighbourhood_concept_plan_json.zip",
				// "http://cosmos.surrey.ca/geo_ref/Images/OpenDataArchives/park_lights_JSON.zip",
				// "http://data.surrey.ca/dataset/229124d3-982c-4e81-919f-d0bfb9eb28a9/resource/739c894b-bf00-45d3-9501-5123a0d15933/download/qsustainabilitycharter-implementationdashboarddata-sheets--open-datacsv-files-consolidated-for-ckanp",
				// "http://cosmos.surrey.ca/geo_ref/Images/OpenDataArchives/town_centre_densities_JSON.zip",
				// "http://cosmos.surrey.ca/geo_ref/Images/OpenDataArchives/traffic_calming_JSON.zip",
				// "http://cosmos.surrey.ca/geo_ref/Images/OpenDataArchives/town_centre_land_use_plans_JSON.zip"];
				// if (missingResources.indexOf(resource.url) >= 0) {
				//     console.log("download resource " + resource.url);
				// } else {
				// 	continue;
				// }

				downloadResource(filepath, resource.url)

	    		if (resource.format === "json") {

	    		}
	    		if (resource.format === "csv") {

	    		}
	    	}
	    	
	    }
  
	});


  })

const requestMetadata = function() {

    for (i = 0; i < 250; i++) {

		var requestLink = 'http://cosmos.surrey.ca/cosrest/rest/services/OpenData/MapServer/_?f=json';
        requestLink = requestLink.replace(/_/i, i.toString());
        console.log("requestLink: " + requestLink);

		requestPromise(requestLink)
			.then( (response) => {
				var responseStr = JSON.stringify(response);

                var filename = "./metadata/_.json";
                filename = filename.replace(/_/i, response.body.id + ' ' + response.body.name);
                console.log("response: " + response.body.name);
				fs.writeFile(filename, responseStr, function(err) {
					if(err) {
						return console.log(err);
					}
				});
				return responseStr;
			}).catch(function(err){
			console.log("error " + err);

			});
    }
}

// node index.js downloadMetadata
program
    .command('downloadMetadata')
    .description('download the metadata for the datasets using URL')
    .action(function(){
        return new Promise((resolve, reject) => {
            requestMetadata((err, result) => {
                if (err) {
                    return reject(err)
                }
                return resolve(result)
            });
        });
    })

program.parse(process.argv);

