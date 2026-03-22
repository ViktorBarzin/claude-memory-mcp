// Tags browser component
function tagsComponent() {
  return {
    tags: [],
    filteredTags: [],
    searchQuery: '',
    selectedTag: null,
    memories: [],
    loading: false,
    memoriesLoading: false,
    expandedId: null,

    async init() {
      this.loading = true;
      try {
        const data = await api.get('/api/tags');
        this.tags = data.tags;
        this.filteredTags = data.tags;
      } catch (e) { console.error('Failed to load tags:', e); }
      this.loading = false;
    },

    filterTags() {
      const q = this.searchQuery.toLowerCase();
      if (!q) {
        this.filteredTags = this.tags;
      } else {
        this.filteredTags = this.tags.filter(t => t.tag.toLowerCase().includes(q));
      }
    },

    async selectTag(tag) {
      if (this.selectedTag === tag) {
        this.selectedTag = null;
        this.memories = [];
        return;
      }
      this.selectedTag = tag;
      this.memoriesLoading = true;
      try {
        const data = await api.get(`/api/memories?tag=${encodeURIComponent(tag)}&limit=100`);
        this.memories = data.memories;
      } catch (e) { console.error('Failed to load memories for tag:', e); }
      this.memoriesLoading = false;
    },

    toggle(id) {
      this.expandedId = this.expandedId === id ? null : id;
    },

    preview(content, len = 100) {
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
