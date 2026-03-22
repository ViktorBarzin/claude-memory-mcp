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
        setTimeout(() => this.renderCharts(), 50);
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
          datasets: [{ data, backgroundColor: colors, borderColor: '#252019', borderWidth: 2 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom', labels: { color: '#c4b8a8', padding: 12 } },
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
            backgroundColor: '#d4a04a',
            borderRadius: 4,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#c4b8a8' }, grid: { color: '#3a3228' } },
            y: { ticks: { color: '#c4b8a8' }, grid: { color: '#3a3228' }, beginAtZero: true },
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
              borderColor: '#7a8b6f',
              backgroundColor: 'rgba(122,139,111,0.1)',
              fill: true,
              tension: 0.3,
            },
            {
              label: 'Updated',
              data: updated,
              borderColor: '#d4a04a',
              backgroundColor: 'rgba(212,160,74,0.1)',
              fill: true,
              tension: 0.3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#c4b8a8', maxTicksLimit: 10 }, grid: { color: '#3a3228' } },
            y: { ticks: { color: '#c4b8a8' }, grid: { color: '#3a3228' }, beginAtZero: true },
          },
          plugins: { legend: { labels: { color: '#c4b8a8' } } },
        },
      });
    },
  };
}

function generateColors(count) {
  const palette = [
    '#d4a04a', '#7a8b6f', '#c4785a', '#8b7355', '#6b8e8e',
    '#b8860b', '#9aad82', '#d4764a', '#7a6b55', '#5f8a8a',
  ];
  const result = [];
  for (let i = 0; i < count; i++) result.push(palette[i % palette.length]);
  return result;
}
