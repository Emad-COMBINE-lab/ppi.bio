function show_not_found(){
    document.getElementById("not_found_indicator").style.display = 'block';
    document.getElementById("error_indicator").style.display = 'none';
    document.getElementById("progress_bar_container").style.display = 'none';
    document.getElementById("loading_indicator").style.display = 'none';
    document.getElementById("queue_indicator").style.display = 'none';
}

function show_error(){
    document.getElementById("not_found_indicator").style.display = 'none';
    document.getElementById("error_indicator").style.display = 'block';
    document.getElementById("progress_bar_container").style.display = 'none';
    document.getElementById("loading_indicator").style.display = 'none';
    document.getElementById("queue_indicator").style.display = 'none';
}

function show_progress(){
    document.getElementById("not_found_indicator").style.display = 'none';
    document.getElementById("error_indicator").style.display = 'none';
    document.getElementById("progress_bar_container").style.display = 'block';
    document.getElementById("loading_indicator").style.display = 'none';
    document.getElementById("queue_indicator").style.display = 'none';
}

function show_queue(){
    document.getElementById("not_found_indicator").style.display = 'none';
    document.getElementById("error_indicator").style.display = 'none';
    document.getElementById("progress_bar_container").style.display = 'none';
    document.getElementById("loading_indicator").style.display = 'block';
    document.getElementById("queue_indicator").style.display = 'block';
}

function refresh() {
    setTimeout(function () {
        location.reload()
    }, 100);
}

async function update_progressbar(result_id) {
    return await fetch('/infer/proteome/task/' + result_id + '.json')
      .then(async function (response) {
          if (response.status === 200 || response.status === 202) {
              let task = await response.json();

              if (task["status"] === "P") {
                  show_progress();
                  const percent_computed = Math.round(100 * task["computed_proteins"] / task["total_proteins"]);
                  document.getElementById("progress_bar").style.width = percent_computed + '%';
                  setTimeout(update_progressbar, 5000, result_id);
              } else if (task["status"] === "Q") {
                  show_queue();
                  setTimeout(update_progressbar, 5000, result_id);
              } else {
                   refresh();
              }
          } else if (response.status === 404) {
              show_not_found();
          } else {
              show_error();
          }
      })

}

document.addEventListener("DOMContentLoaded", async function () {
  const task_id = document.head.querySelector("[property~=data-task-id][content]").content;
  setTimeout(update_progressbar, 5000, task_id);
});
