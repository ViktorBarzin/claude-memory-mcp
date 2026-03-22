// Dashboard stats component
function dashboardComponent() {
  return {
    stats: null,
    loading: false,
    charts: {},

    async init() {
      this.loading = true;
      try {
        this.stats = await api.get('/api/stats');
        this.$nextTick(() => this.renderCharts());
      } catch (e) {
        console.error('Failed to load stats:', e);
      }
      this.loading = false;
    },

    renderCharts() {
      if (!this.stats) return;
      this.renderCategoryChart();
      this.renderImportanceChart();
      this.renderActivityChart();
    },

    renderCategoryChart() {
      const ctx = this.$refs.categoryChart;
      if (!ctx) return;
      if (this.charts.category) this.charts.category.destroy();

      const labels = Object.keys(this.stats.by_category);
      const data = Object.values(this.stats.by_category);
      const colors = generateColors(labels.length);

      this.charts.category = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{ data, backgroundColor: colors, borderColor: '#1e293b', borderWidth: 2 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 12 } },
          },
        },
      });
    },

    renderImportanceChart() {
      const ctx = this.$refs.importanceChart;
      if (!ctx) return;
      if (this.charts.importance) this.charts.importance.destroy();

      const labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'];
      const data = labels.map(l => this.stats.by_importance[l] || 0);

      this.charts.importance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Memories',
            data,
            backgroundColor: '#6366f1',
            borderRadius: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
            y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, beginAtZero: true },
          },
          plugins: { legend: { display: false } },
        },
      });
    },

    renderActivityChart() {
      const ctx = this.$refs.activityChart;
      if (!ctx) return;
      if (this.charts.activity) this.charts.activity.destroy();

      const activity = this.stats.recent_activity || [];
      const labels = activity.map(a => a.date);
      const created = activity.map(a => a.created);
      const updated = activity.map(a => a.updated);

      this.charts.activity = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Created',
              data: created,
              borderColor: '#22c55e',
              backgroundColor: 'rgba(34,197,94,0.1)',
              fill: true,
              tension: 0.3,
            },
            {
              label: 'Updated',
              data: updated,
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245,158,11,0.1)',
              fill: true,
              tension: 0.3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 }, grid: { color: '#334155' } },
            y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, beginAtZero: true },
          },
          plugins: { legend: { labels: { color: '#94a3b8' } } },
        },
      });
    },
  };
}

function generateColors(count) {
  const palette = [
    '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
    '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#64748b',
  ];
  const result = [];
  for (let i = 0; i < count; i++) result.push(palette[i % palette.length]);
  return result;
}
