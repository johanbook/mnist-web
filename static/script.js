var plot = document.getElementById("plot");

var canvas = document.getElementById("paint");
var width = canvas.width;
var height = canvas.height;

var context = canvas.getContext("2d");

var currentX, currentY, previousX, previousY;
var hold = false;
context.lineWidth = 18;

var fill_value = true;
var stroke_value = false;

function reset() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  connect_python(
    "/imshow",
    (image) => (imshow.src = URL.createObjectURL(image))
  );
  connect_python_json("/predict", (prediction) => update_chart(prediction));
}
reset();

function connect_python(endpoint, callback) {
  const formData = new FormData();
  formData.append("image", canvas.toDataURL());
  fetch(endpoint, {
    method: "POST",
    body: formData,
  })
    .then((response) => response.blob())
    .then((image) => callback(image));
}

function connect_python_json(endpoint, callback) {
  const formData = new FormData();
  formData.append("image", canvas.toDataURL());
  fetch(endpoint, {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => callback(data));
}

function pencil() {
  canvas.onmousedown = function (e) {
    currentX = e.clientX - canvas.offsetLeft;
    currentY = e.clientY - canvas.offsetTop;
    hold = true;

    previousX = currentX;
    previousY = currentY;
    context.beginPath();
    context.moveTo(previousX, previousY);
  };

  canvas.onmousemove = function (e) {
    if (hold) {
      currentX = e.clientX - canvas.offsetLeft;
      currentY = e.clientY - canvas.offsetTop;
      draw();
    }
  };

  canvas.onmouseup = function (e) {
    hold = false;

    connect_python("/imshow", (image) => {
      imshow.src = URL.createObjectURL(image);
    });
    connect_python_json("/predict", (prediction) => update_chart(prediction));
  };

  canvas.onmouseout = function (e) {
    hold = false;
  };

  function draw() {
    context.lineTo(currentX, currentY);
    context.stroke();
  }
}

const barchart_options = {
  scales: {
    xAxes: [
      {
        gridLines: { display: false },
        scaleLabel: { display: true, labelString: "Digit" },
      },
    ],
    yAxes: [
      {
        gridLines: { display: false },
        scaleLabel: { display: true, labelString: "Probability" },
        ticks: { min: 0, max: 1 },
      },
    ],
  },
  legend: { display: false },
  responsive: false,
  tooltips: {
    callbacks: {
      label: function (tooltipItem) {
        return Number(100 * tooltipItem.yLabel).toFixed(2) + "%";
      },
    },
  },
};

var barchart;
function update_chart(prediction) {
  var ctx = document.getElementById("barchart").getContext("2d");
  if (barchart) barchart.destroy();
  barchart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: prediction.labels,
      datasets: [
        {
          label: "Probability",
          data: prediction.data,
          borderWidth: 1,
          borderColor: "gray",
          backgroundColor: "#17a2b8",
          hoverBackgroundColor: "#19b4cc",
        },
      ],
    },
    options: barchart_options,
  });
}
