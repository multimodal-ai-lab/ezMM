// Small progressive enhancements for the ezMM web UI. Everything works without JS.

(function () {
    const root = document.documentElement;
    const toast = document.getElementById("toast");
    let toastTimer;

    function showToast(message) {
        if (!toast) return;
        toast.textContent = message;
        toast.classList.add("show");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => toast.classList.remove("show"), 1600);
    }

    // Theme toggle (remembered per browser)
    const themeToggle = document.getElementById("theme-toggle");
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const current = root.dataset.theme ||
                (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
            const next = current === "dark" ? "light" : "dark";
            root.dataset.theme = next;
            try { localStorage.setItem("ezmm-theme", next); } catch (e) {}
        });
    }

    // Copy buttons
    document.addEventListener("click", async (event) => {
        const button = event.target.closest("[data-copy]");
        if (!button) return;
        event.preventDefault();
        try {
            await navigator.clipboard.writeText(button.dataset.copy);
            showToast("Copied " + (button.dataset.copy.length > 24 ? "to clipboard" : button.dataset.copy));
        } catch (e) {
            showToast("Copy failed");
        }
    });

    // Press "/" to focus the search
    const search = document.getElementById("search");
    document.addEventListener("keydown", (event) => {
        if (event.key === "/" && search && document.activeElement !== search &&
            !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
            event.preventDefault();
            search.focus();
            search.select();
        }
    });

    // Search with a file: submit on choose or drop
    const dropzone = document.getElementById("dropzone");
    if (dropzone) {
        const input = dropzone.querySelector("input[type=file]");
        const submit = () => {
            dropzone.classList.add("busy");
            dropzone.querySelector("strong").textContent = "Searching with " + input.files[0].name + "…";
            dropzone.submit();
        };
        input.addEventListener("change", () => input.files.length && submit());
        ["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
            event.preventDefault();
            dropzone.classList.add("dragover");
        }));
        ["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, () => dropzone.classList.remove("dragover")));
        dropzone.addEventListener("drop", (event) => {
            event.preventDefault();
            if (!event.dataTransfer.files.length) return;
            input.files = event.dataTransfer.files;
            submit();
        });
    }

    // Preview videos on hover (delegated, so it also works for cards loaded later)
    ["mouseenter", "mouseleave"].forEach((type) => document.addEventListener(type, (event) => {
        const video = event.target.matches?.(".card") && event.target.querySelector("video");
        if (!video) return;
        if (type === "mouseenter") video.play().catch(() => {});
        else video.pause();
    }, true));

    // Endless scrolling: append the next page of cards when the end of the grid comes into view
    const grid = document.querySelector(".grid[data-next]");
    if (grid && grid.dataset.next && "IntersectionObserver" in window) {
        const pager = document.querySelector(".pagination");
        if (pager) pager.style.display = "none";
        const loader = document.createElement("div");
        loader.className = "loader muted small";
        loader.innerHTML = '<span class="spinner" aria-hidden="true"></span> Loading more…';
        grid.after(loader);
        let loading = false;

        const observer = new IntersectionObserver(async (entries) => {
            if (loading || !entries.some((entry) => entry.isIntersecting)) return;
            loading = true;
            try {
                const url = new URL(grid.dataset.next, location.href);
                url.searchParams.set("partial", "1");
                const response = await fetch(url);
                if (!response.ok) throw new Error(response.statusText);
                const html = new DOMParser().parseFromString(await response.text(), "text/html");
                const page = html.querySelector(".grid[data-next]");
                const shown = new Set([...grid.children].map((card) => card.getAttribute("href")));
                [...page.children].forEach((card) => shown.has(card.getAttribute("href")) || grid.append(card));
                grid.dataset.next = page.dataset.next;
            } catch (e) {
                grid.dataset.next = "";
                if (pager) pager.style.display = "";
                showToast("Could not load more items");
            }
            loading = false;
            observer.unobserve(loader);
            if (grid.dataset.next) observer.observe(loader);  // Re-checks if the loader is still in view
            else loader.remove();
        }, {rootMargin: "600px 0px"});
        observer.observe(loader);
    }
})();
