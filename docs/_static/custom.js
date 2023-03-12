let sidebar_scroll_element = document.querySelector(".sidebar-scroll");

let saved_top = sessionStorage.getItem("sidebar-scroll-top");
if (saved_top !== null) {
  sidebar_scroll_element.scrollTop = parseInt(saved_top, 10);
}

window.addEventListener("beforeunload", () => {
  sessionStorage.setItem("sidebar-scroll-top", sidebar_scroll_element.scrollTop);
});
