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
    // Add memory
    showAddForm: false,
    addForm: { content: '', category: 'facts', tags: '', importance: 0.5 },
    // Share memory
    allUsers: [],
    showShareForm: null,
    shareForm: { user: '', permission: 'read' },
    shareUserFilter: '',
    // Tag shares
    tagShares: [],
    showTagShareForm: false,
    tagShareForm: { tag: '', user: '', permission: 'read' },
    tagShareUserFilter: '',
    errorMsg: '',

    async init() {
      await Promise.all([this.loadMemories(), this.loadShared(), this.loadCategories(), this.loadUsers(), this.loadTagShares()]);
    },

    async loadCategories() {
      try {
        const data = await api.get('/api/categories');
        this.categories = data.categories;
      } catch (e) { console.error('Failed to load categories:', e); }
    },

    async loadUsers() {
      try {
        const data = await api.get('/api/users');
        this.allUsers = data.users;
      } catch (e) { console.error('Failed to load users:', e); }
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
      this.showShareForm = null;
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
      this.errorMsg = '';
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
        this.errorMsg = 'Save failed: ' + e.message;
      }
    },

    async confirmDelete(id) {
      this.deleteConfirm = id;
    },

    async doDelete(id) {
      this.errorMsg = '';
      try {
        await api.del(`/api/memories/${id}`);
        this.deleteConfirm = null;
        this.offset = 0;
        await this.loadMemories();
      } catch (e) {
        this.errorMsg = 'Delete failed: ' + e.message;
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

    // Add memory
    resetAddForm() {
      this.addForm = { content: '', category: 'facts', tags: '', importance: 0.5 };
      this.showAddForm = false;
    },

    get addFormCharCount() {
      return (this.addForm.content || '').length;
    },

    async addMemory() {
      if (!this.addForm.content.trim()) return;
      this.errorMsg = '';
      try {
        await api.post('/api/memories', {
          content: this.addForm.content,
          category: this.addForm.category,
          tags: this.addForm.tags,
          importance: parseFloat(this.addForm.importance),
        });
        this.resetAddForm();
        this.offset = 0;
        await Promise.all([this.loadMemories(), this.loadCategories()]);
      } catch (e) {
        this.errorMsg = 'Failed to add memory: ' + e.message;
      }
    },

    // Share memory
    openShareForm(memId) {
      this.showShareForm = memId;
      this.shareForm = { user: '', permission: 'read' };
      this.shareUserFilter = '';
    },

    closeShareForm() {
      this.showShareForm = null;
    },

    get filteredUsers() {
      const q = this.shareForm.user.toLowerCase();
      if (!q) return this.allUsers.slice(0, 10);
      return this.allUsers.filter(u => u.toLowerCase().includes(q));
    },

    selectShareUser(user) {
      this.shareForm.user = user;
    },

    async shareMemory(memId) {
      if (!this.shareForm.user.trim()) return;
      this.errorMsg = '';
      try {
        await api.post(`/api/memories/${memId}/share`, {
          shared_with: this.shareForm.user,
          permission: this.shareForm.permission,
        });
        this.closeShareForm();
        // Refresh shares if expanded
        if (this.sharesExpanded === memId) {
          await this.toggleShares(memId);
          await this.toggleShares(memId);
        }
      } catch (e) {
        this.errorMsg = 'Share failed: ' + e.message;
      }
    },

    async loadTagShares() {
      try {
        const data = await api.get('/api/memories/my-shares');
        this.tagShares = data.tag_shares || [];
      } catch (e) { console.error('Failed to load tag shares:', e); }
    },

    get filteredTagShareUsers() {
      const q = this.tagShareForm.user.toLowerCase();
      if (!q) return this.allUsers.slice(0, 10);
      return this.allUsers.filter(u => u.toLowerCase().includes(q));
    },

    selectTagShareUser(user) {
      this.tagShareForm.user = user;
    },

    async addTagShare() {
      if (!this.tagShareForm.tag.trim() || !this.tagShareForm.user.trim()) return;
      this.errorMsg = '';
      try {
        await api.post('/api/memories/share-tag', {
          tag: this.tagShareForm.tag,
          shared_with: this.tagShareForm.user,
          permission: this.tagShareForm.permission,
        });
        this.tagShareForm = { tag: '', user: '', permission: 'read' };
        this.showTagShareForm = false;
        await this.loadTagShares();
      } catch (e) {
        this.errorMsg = 'Failed to share tag: ' + e.message;
      }
    },

    async removeTagShare(tag, sharedWith) {
      this.errorMsg = '';
      try {
        await api.del('/api/memories/share-tag', { tag, shared_with: sharedWith });
        await this.loadTagShares();
      } catch (e) {
        this.errorMsg = 'Failed to remove tag share: ' + e.message;
      }
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
