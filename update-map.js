const {spawn} = require('child_process');
var sql = require('mssql');
const fs = require('fs')

var dbConfig = {
  server: "outagesnyc.database.windows.net",
  database: "main",
  user: "outagesnycuser",
  password: "VanHiggins22@",
  port: 1433,
  options: {
    encrypt: true
  }
};

var conn = new sql.ConnectionPool(dbConfig);

function updateMapObjects() {
  fs.readFile('./USA-road-d.E.co', 'utf8' , (err, data) => {
    if (err) {
      console.error(err)
      return
    }
    let sentences = [];
    for (let i = 0; i < data.length; i++) {
      let sentence = ""
      while (data[i] !== 'c' && data[i] !== 'v') {
        sentence += data[i];
        i++;
      }
      let words = sentence.split(' ');
      if (words.length === 4) {
        words[3] = words[3].substr(0, words[3].length - 2);
        console.log(Number(words[1]), Number(words[2]) / 1000000.0, Number(words[3]) / 100000.0)
        conn.connect().then(function() {
          var req = new sql.Request(conn);
            req.input('id', sql.Int, Number(words[1]));
            req.input('latitude', sql.Decimal(30, 15), Number(words[2]) / 1000000.0);
            req.input('longitude', sql.Decimal(30, 15), Number(words[3]) / 100000.0);
            req.query('insert into routing_intersection_data values (@id, @latitude, @longitude)')
              .then((r, e) => console.log(r, e));
        }).then((rows, err) => console.log(err))
      }
    }
  })
}

//crawlData()
updateMapObjects()
// "https://maps.googleapis.com/maps/api/geocode/json?address=147+MADISON+ST&key=AIzaSyDoqw7Hgx2yj0Hiqbziov8cKDL9uFm8Nz4"