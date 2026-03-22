// Memory browser/editor component
function memoriesBrowser() {
  return {
    memories: [],
    sharedMemories: [],
    categories: [],
    selectedCategory: '',
    total: 0,
    offset: 0,
    limit: 50,
    loading: false,
    expandedId: null,
    editingId: null,
    editForm: { content: '', tags: '', importance: 0.5 },
    sharesExpanded: null,
    sharesData: [],
    deleteConfirm: null,

    async init() {
      await Promise.all([this.loadMemories(), this.loadShared(), this.loadCategories()]);
    },

    async loadCategories() {
      try {
        const data = await api.get('/api/categories');
        this.categories = data.categories;
      } catch (e) { console.error('Failed to load categories:', e); }
    },

    async loadMemories() {
      this.loading = true;
      try {
        let url = `/api/memories?limit=${this.limit}&offset=${this.offset}`;
        if (this.selectedCategory) url += `&category=${encodeURIComponent(this.selectedCategory)}`;
        const data = await api.get(url);
        if (this.offset === 0) {
          this.memories = data.memories;
        } else {
          this.memories = [...this.memories, ...data.memories];
        }
        this.total = data.total;
      } catch (e) { console.error('Failed to load memories:', e); }
      this.loading = false;
    },

    async loadShared() {
      try {
        const data = await api.get('/api/memories/shared-with-me');
        this.sharedMemories = data.memories;
      } catch (e) { console.error('Failed to load shared:', e); }
    },

    async filterByCategory() {
      this.offset = 0;
      await this.loadMemories();
    },

    async loadMore() {
      this.offset += this.limit;
      await this.loadMemories();
    },

    get hasMore() {
      return this.memories.length < this.total;
    },

    toggle(id) {
      this.expandedId = this.expandedId === id ? null : id;
      this.editingId = null;
      this.sharesExpanded = null;
    },

    startEdit(mem) {
      this.editingId = mem.id;
      this.editForm = {
        content: mem.content,
        tags: mem.tags || '',
        importance: mem.importance,
      };
    },

    cancelEdit() {
      this.editingId = null;
    },

    async saveEdit(id) {
      try {
        await api.put(`/api/memories/${id}`, {
          content: this.editForm.content,
          tags: this.editForm.tags,
          importance: parseFloat(this.editForm.importance),
        });
        this.editingId = null;
        this.offset = 0;
        await this.loadMemories();
      } catch (e) {
        alert('Save failed: ' + e.message);
      }
    },

    async confirmDelete(id) {
      this.deleteConfirm = id;
    },

    async doDelete(id) {
      try {
        await api.del(`/api/memories/${id}`);
        this.deleteConfirm = null;
        this.offset = 0;
        await this.loadMemories();
      } catch (e) {
        alert('Delete failed: ' + e.message);
      }
    },

    async toggleShares(memId) {
      if (this.sharesExpanded === memId) {
        this.sharesExpanded = null;
        return;
      }
      try {
        const data = await api.get('/api/memories/my-shares');
        this.sharesData = data.memory_shares.filter(s => s.memory_id === memId);
        this.sharesExpanded = memId;
      } catch (e) { console.error('Failed to load shares:', e); }
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
