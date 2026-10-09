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

    // Preview videos on hover
    document.querySelectorAll(".card video").forEach((video) => {
        const card = video.closest(".card");
        card.addEventListener("mouseenter", () => video.play().catch(() => {}));
        card.addEventListener("mouseleave", () => video.pause());
    });
})();
