// API client with Bearer auth
const api = {
  getToken() {
    return sessionStorage.getItem('api_key');
  },

  getUserId() {
    return sessionStorage.getItem('user_id');
  },

  setAuth(key, userId) {
    sessionStorage.setItem('api_key', key);
    sessionStorage.setItem('user_id', userId);
  },

  clearAuth() {
    sessionStorage.removeItem('api_key');
    sessionStorage.removeItem('user_id');
  },

  isAuthenticated() {
    return !!this.getToken();
  },

  async fetch(url, options = {}) {
    const token = this.getToken();
    if (!token) {
      window.dispatchEvent(new CustomEvent('auth:required'));
      throw new Error('Not authenticated');
    }

    const headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      ...options.headers,
    };

    const res = await fetch(url, { ...options, headers });

    if (res.status === 401) {
      this.clearAuth();
      window.dispatchEvent(new CustomEvent('auth:required'));
      throw new Error('Unauthorized');
    }

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`API error ${res.status}: ${body}`);
    }

    return res.json();
  },

  get(url) {
    return this.fetch(url);
  },

  post(url, data) {
    return this.fetch(url, { method: 'POST', body: JSON.stringify(data) });
  },

  put(url, data) {
    return this.fetch(url, { method: 'PUT', body: JSON.stringify(data) });
  },

  del(url) {
    return this.fetch(url, { method: 'DELETE' });
  },

  async login(key) {
    const res = await fetch('/api/auth-check', {
      headers: { 'Authorization': `Bearer ${key}` },
    });
    if (!res.ok) throw new Error('Invalid API key');
    const data = await res.json();
    this.setAuth(key, data.user_id);
    return data;
  },

  logout() {
    this.clearAuth();
    window.dispatchEvent(new CustomEvent('auth:required'));
  },
};
