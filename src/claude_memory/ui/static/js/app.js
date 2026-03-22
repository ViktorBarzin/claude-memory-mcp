// Alpine.js global store and tab navigation
document.addEventListener('alpine:init', () => {
  Alpine.store('app', {
    activeTab: 'memories',
    userId: api.getUserId() || '',
    authenticated: api.isAuthenticated(),
    loginKey: '',
    loginError: '',
    loginLoading: false,

    async login() {
      this.loginError = '';
      this.loginLoading = true;
      try {
        const data = await api.login(this.loginKey);
        this.userId = data.user_id;
        this.authenticated = true;
        this.loginKey = '';
      } catch (e) {
        this.loginError = e.message;
      } finally {
        this.loginLoading = false;
      }
    },

    logout() {
      api.logout();
      this.authenticated = false;
      this.userId = '';
    },

    switchTab(tab) {
      this.activeTab = tab;
    },
  });

  window.addEventListener('auth:required', () => {
    Alpine.store('app').authenticated = false;
    Alpine.store('app').userId = '';
  });
});
