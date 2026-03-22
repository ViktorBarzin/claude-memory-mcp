// Search interface component
function searchComponent() {
  return {
    query: '',
    category: '',
    sortBy: 'importance',
    results: [],
    categories: [],
    loading: false,
    expandedId: null,
    debounceTimer: null,

    async init() {
      try {
        const data = await api.get('/api/categories');
        this.categories = data.categories;
      } catch (e) { console.error(e); }
    },

    onInput() {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => this.doSearch(), 300);
    },

    async doSearch() {
      if (!this.query.trim()) {
        this.results = [];
        return;
      }
      this.loading = true;
      try {
        const body = {
          context: this.query,
          sort_by: this.sortBy,
          limit: 20,
        };
        if (this.category) body.category = this.category;
        const data = await api.post('/api/memories/recall', body);
        this.results = data.memories;
      } catch (e) {
        console.error('Search failed:', e);
      }
      this.loading = false;
    },

    toggle(id) {
      this.expandedId = this.expandedId === id ? null : id;
    },

    preview(content, len = 120) {
      if (!content) return '';
      return content.length > len ? content.substring(0, len) + '...' : content;
    },

    relativeTime(iso) {
      const diff = Date.now() - new Date(iso).getTime();
      const mins = Math.floor(diff / 60000);
      if (mins < 1) return 'just now';
      if (mins < 60) return `${mins}m ago`;
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return `${hrs}h ago`;
      const days = Math.floor(hrs / 24);
      if (days < 30) return `${days}d ago`;
      return new Date(iso).toLocaleDateString();
    },
  };
}
