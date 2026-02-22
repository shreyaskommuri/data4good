import jsPDF from 'jspdf';

/**
 * Compute dynamic budget allocation across 6 categories based on tract data
 */
function generateBudgetAllocation(tract, params, simData) {
  const ejBurden     = tract.ej_percentile    || 0;   // 0–100
  const coastalJobs  = tract.coastal_jobs_pct || 0;   // 0–100
  const medianIncome = tract.median_income     || 50000;
  const exodusProb   = tract.exodus_prob       || 0;
  const recoveryTime = Math.round(simData?.recovery_time || 30);
  const budget       = simData?.emergency_fund || 10_000_000;

  // Normalized 0–1 helpers (clamped so no term goes negative)
  const incomeVuln  = Math.max(0, 1 - medianIncome / 120000); // 1 = poorest, 0 = high income
  const ejFrac      = ejBurden  / 100;
  const cjFrac      = coastalJobs / 100;
  const rtFrac      = Math.min(1, recoveryTime / 90);
  const epFrac      = Math.min(1, exodusProb * 10);

  // Raw weights — all terms non-negative, minimum floor of 0.05 so every category always gets something
  const clamp = (v) => Math.max(0.05, v);
  const w = {
    'Emergency Cash Assistance':   clamp(incomeVuln * 0.9 + ejFrac * 0.5),
    'Workforce Retraining':        clamp(cjFrac * 1.2 + params.severity * 0.4),
    'Business Recovery Loans':     clamp(epFrac * 1.0 + params.severity * 0.5),
    'Infrastructure & Utilities':  clamp(rtFrac  * 1.0 + params.severity * 0.3),
    'Housing Stabilization':       clamp(incomeVuln * 0.7 + ejFrac * 0.4),
    'Community Outreach & Equity': clamp(ejFrac * 1.1 + incomeVuln * 0.3),
  };

  const totalW = Object.values(w).reduce((a, b) => a + b, 0);

  return Object.entries(w).map(([name, weight]) => {
    const pct    = weight / totalW;
    const amount = Math.round(budget * pct / 1000) * 1000; // round to nearest $1k
    return { name, pct, amount };
  });
}

/**
 * Generate policy rationale for each budget category
 */
function policyRationale(category, tract, params, simData) {
  const ej     = tract.ej_percentile    || 0;
  const cj     = tract.coastal_jobs_pct || 0;
  const income = tract.median_income     || 50000;
  const ep     = (tract.exodus_prob || 0) * 100;
  const rt     = Math.round(simData?.recovery_time || 30);

  const map = {
    'Emergency Cash Assistance':
      `Median household income of $${income.toLocaleString()} leaves residents with little financial buffer. Direct cash transfers and expanded CalWORKs/SNAP enrollment will prevent evictions and hunger in the first 30 days post-event.`,
    'Workforce Retraining':
      `${cj.toFixed(1)}% of jobs are in coastal sectors vulnerable to disruption. Rapid partnerships with Santa Barbara City College and the Regional Occupational Center should deliver 4–8 week certifications in healthcare, logistics, and remote-work roles.`,
    'Business Recovery Loans':
      `With a ${ep.toFixed(1)}% population exodus probability and severity of ${(params.severity * 100).toFixed(0)}%, small business survival is critical to retaining the tax base. Low-interest bridge loans (2–3%) and deferred repayment terms prevent permanent closures.`,
    'Infrastructure & Utilities':
      `Projected recovery time of ${rt} days indicates sustained infrastructure stress. Priority restoration of power, water, and transit corridors directly accelerates workforce re-engagement and reduces downstream economic losses.`,
    'Housing Stabilization':
      `An EJ burden of ${ej}% and low median income signal high renter vulnerability. Emergency rental assistance, code-compliance repair grants, and temporary relocation vouchers will prevent displacement and preserve community cohesion.`,
    'Community Outreach & Equity':
      `With an EJ percentile of ${ej}%, culturally responsive outreach is essential. Bilingual navigators, mobile assistance centers, and trusted community-based organization partnerships ensure equitable program uptake.`,
  };
  return map[category] || '';
}

/**
 * Generate county-specific action items based on tract characteristics
 */
function generateActionItems(tract, params, simData) {
  const ejBurden = tract.ej_percentile || 0;
  const coastalJobs = tract.coastal_jobs_pct || 0;
  const medianIncome = tract.median_income || 50000;
  const recoveryTime = Math.round(simData?.recovery_time || 30);
  const severity = params.severity;
  const riskLevel = tract.exodus_prob > 0.08 ? 'critical' : 
                   tract.exodus_prob > 0.06 ? 'high' : 
                   tract.exodus_prob > 0.04 ? 'moderate' : 'low';

  const actions = [];

  // Action 1: Based on EJ burden
  if (ejBurden > 75) {
    actions.push(`This tract has an Environmental Justice burden percentile of ${ejBurden}%, indicating significant vulnerability to climate and economic shocks. Prioritize engagement with community organizations and ensure that any recovery assistance programs are linguistically accessible and culturally responsive. Consider partnering with local nonprofits already working in this community to build trust and ensure equitable distribution of resources.`);
  } else if (ejBurden > 50) {
    actions.push(`With an Environmental Justice burden percentile of ${ejBurden}%, this tract requires enhanced monitoring and resource allocation. Establish regular check-ins with community leaders to understand specific workforce challenges and barriers to employment. Coordinate with county social services to identify families that may need additional support during economic recovery.`);
  } else {
    actions.push(`This tract has moderate environmental justice considerations (${ejBurden}% percentile). While less burdened than some areas, proactive workforce development programs can still strengthen community resilience. Focus on skills training aligned with post-storm employment opportunities.`);
  }

  // Action 2: Based on coastal jobs dependency
  if (coastalJobs > 40) {
    actions.push(`With ${coastalJobs.toFixed(1)}% of employment in coastal sectors, this tract faces significant disruption from coastal events. Develop rapid retraining programs targeting alternative industries within 7 days of impact. Coordinate with state workforce development agencies and the Santa Barbara County Economic Development Board to identify emerging job opportunities in agriculture, tourism support, healthcare, and remote work sectors. Consider establishing temporary subsidized employment programs to bridge workers to new sectors.`);
  } else if (coastalJobs > 20) {
    actions.push(`Approximately ${coastalJobs.toFixed(1)}% of employment is in coastal sectors, creating moderate economic vulnerability. Begin workforce diversification planning now by connecting residents with training opportunities in growth sectors. Partner with Santa Barbara City College and the Regional Occupational Center to develop sector-specific training that can activate quickly if needed.`);
  } else {
    actions.push(`With ${coastalJobs.toFixed(1)}% coastal sector employment, this tract has greater economic diversity. Focus capacity-building efforts on underemployed residents through career pathway programs and skills certification courses in high-wage sectors.`);
  }

  // Action 3: Based on recovery time and income
  if (recoveryTime > 60 && medianIncome < 75000) {
    actions.push(`The projected recovery time of ${recoveryTime} days combined with a median income of $${medianIncome.toLocaleString()} suggests residents will face acute financial hardship during recovery. Establish an emergency financial assistance program targeting renters and small business owners. Partner with nonprofit lenders like Santa Cruz County Bank and SCORE mentors to provide low-interest bridge loans. Coordinate with the county to temporarily expand welfare programs (SNAP, CalWORKs) to meet surge demand during peak recovery period.`);
  } else if (recoveryTime > 30) {
    actions.push(`With an expected recovery period of ${recoveryTime} days, residents will require sustained income support. Work with the state Department of Labor to fast-track Unemployment Insurance claims and extend maximum benefit weeks if necessary. Market the Governor's Disaster Assistance and Temporary Housing programs heavily in this tract. Ensure clear communication about eligibility and application processes through community organizations and local media.`);
  } else {
    actions.push(`The relatively short projected recovery time of ${recoveryTime} days suggests focused, shorter-term intervention will be most effective. Maintain emergency services and ensure rapid restoration of utilities and transportation infrastructure, which will dramatically accelerate economic recovery and workforce re-engagement.`);
  }

  // Action 4: Based on risk level
  if (riskLevel === 'critical') {
    actions.push(`This tract faces a critical exodus risk level, indicating substantial out-migration pressure. Implement a comprehensive retention program including: negotiated employer agreements to maintain wages during recovery, flexible work arrangements to support workers managing household disruption, and relocation assistance for those whose homes are damaged. Engage employers with operations in this tract to develop contingency workforce plans before a disaster strikes. This proactive approach can prevent cascading job losses.`);
  } else if (riskLevel === 'high') {
    actions.push(`As a high-risk tract for worker exodus, establish strong employer relationships now to lock in recovery commitments. Create a "Business Continuity Network" for key employers to coordinate workforce retention strategies, cross-training, and shared resources. After a disaster, activate rapid workforce agreements that stabilize income and maintain employment relationships even if physical worksites are temporarily non-operational.`);
  } else {
    actions.push(`This tract has manageable exodus risk, but should still prepare workforce continuity plans. Conduct scenario exercises with major employers to ensure they understand disruption risks and have backup plans for remote work, alternative sites, or accelerated hiring of displaced workers from other areas.`);
  }

  return actions;
}

/**
 * Generate a single-page PDF report for a specific census tract
 * Focused on one tract with current parameters
 */
export async function generatePDFReport(params, simData, selectedTract) {
  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4'
  });

  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 12;
  const contentWidth = pageWidth - margin * 2;
  const headerHeight = 32;
  const footerHeight = 10;
  const footerY = pageHeight - footerHeight;

  // Color scheme (matches UI dark theme)
  const colors = {
    blue: [59, 130, 246],          // #3b82f6
    cyan: [34, 211, 238],          // #22d3ee
    emerald: [52, 211, 153],       // #34d399
    rose: [244, 63, 94],           // #f43f5e
    purple: [167, 139, 250],       // #a78bfa
    text: [232, 232, 240],         // #e8e8f0
    secondary: [139, 140, 160],    // #8b8ca0
    muted: [90, 91, 112]           // #5a5b70
  };

  if (!selectedTract) {
    alert('Please select a tract first');
    return;
  }

  const riskLevel = selectedTract.exodus_prob > 0.08 ? 'CRITICAL' : 
                   selectedTract.exodus_prob > 0.06 ? 'HIGH' : 
                   selectedTract.exodus_prob > 0.04 ? 'MODERATE' : 'LOW';
  const riskColor = selectedTract.exodus_prob > 0.08 ? colors.rose : 
                   selectedTract.exodus_prob > 0.06 ? [255, 165, 0] : 
                   selectedTract.exodus_prob > 0.04 ? colors.purple : colors.emerald;

  const tractMetrics = [
    ['Population', `${(selectedTract.population || 0).toLocaleString()}`],
    ['EJ Burden', `${selectedTract.ej_percentile || 0}%`],
    ['Median Income', `$${(selectedTract.median_income || 0).toLocaleString()}`],
    ['Coastal Jobs', `${selectedTract.coastal_jobs_pct || 0}%`],
  ];

  const params_display = [
    { label: 'Storm Severity', value: `${(params.severity * 100).toFixed(0)}%` },
    { label: 'Duration', value: `${params.duration} days` },
    { label: 'Recovery Rate (r)', value: params.r.toFixed(3) },
    { label: 'Max Employment (K)', value: `${(params.K * 100).toFixed(1)}%` },
  ];

  const metrics_display = simData ? [
    { label: 'Recovery Time', value: `${Math.round(simData.recovery_time || 0)} days`, color: colors.cyan },
    { label: 'Min Employment', value: `${((simData.min_labor_force || 0) * 100).toFixed(1)}%`, color: colors.rose },
    { label: 'Resilience Score', value: `${(simData.resilience_score || 0).toFixed(2)}/1.0`, color: colors.emerald },
    { label: 'Exodus Risk', value: `${(selectedTract.exodus_prob * 100).toFixed(1)}%`, color: colors.purple },
  ] : [];

  // ==== PRE-CALCULATE HEIGHTS ====
  // Fixed section heights (label + divider):
  //   Each section divider+header: 1 (line) + 4 (gap) + 5 (font) + 3 (gap after) = ~9mm
  //   Metric grid (2x2 rows): 2 rows × metricLineH
  const metricLineH = 6;
  const sectionHeaderH = 10; // divider line + section title
  const metricsGridH = 2 * metricLineH; // 2 rows

  // Heights of fixed sections (not counting inter-section gaps):
  const tractHeaderH = 10;           // tract name + risk badge row
  const tractMetricsH = metricsGridH; // 2 rows of metrics
  const scenarioHeaderH = sectionHeaderH;
  const scenarioMetricsH = metricsGridH;
  const resilienceHeaderH = sectionHeaderH;
  const resilienceMetricsH = metricsGridH;
  const actionSectionHeaderH = sectionHeaderH;

  // Pre-calculate action item text splits
  const actionItems = generateActionItems(selectedTract, params, simData);
  const actionFontSize = 7.8;
  const actionLineH = 3.6;
  const actionParaGap = 2.5;
  pdf.setFontSize(actionFontSize);
  const splitActions = actionItems.map(a => pdf.splitTextToSize(a, contentWidth - 6));
  const totalActionTextH = splitActions.reduce((sum, lines) => sum + lines.length * actionLineH, 0)
    + (actionItems.length - 1) * actionParaGap;

  // Total fixed content height (sum of all sections, without inter-section gaps)
  const fixedContentH = tractHeaderH + tractMetricsH
    + scenarioHeaderH + scenarioMetricsH
    + resilienceHeaderH + resilienceMetricsH
    + actionSectionHeaderH + totalActionTextH;

  // Available vertical space for content + inter-section gaps
  const contentStartY = headerHeight + 4;
  const availableH = footerY - contentStartY;

  // Distribute remaining space as equal gaps between the 4 major section blocks:
  // (1) tract info, (2) scenario, (3) resilience, (4) action items
  const numGaps = 4;
  const extraH = Math.max(0, availableH - fixedContentH);
  const sectionGap = extraH / numGaps;

  // ==== BACKGROUND ====
  pdf.setFillColor(10, 11, 15);
  pdf.rect(0, 0, pageWidth, pageHeight, 'F');

  // ==== HEADER BAR ====
  pdf.setFillColor(...colors.blue);
  pdf.rect(0, 0, pageWidth, headerHeight, 'F');

  pdf.setTextColor(232, 232, 240);
  pdf.setFontSize(22);
  pdf.setFont('helvetica', 'bold');
  pdf.text('RESILIENCE REPORT', margin + 5, 13);

  pdf.setFontSize(9);
  pdf.setFont('helvetica', 'normal');
  pdf.setTextColor(...colors.cyan);
  pdf.text(`Santa Barbara County • ${new Date().toLocaleDateString()}`, margin + 5, 22);

  let y = contentStartY;

  // ==== TRACT HEADER ====
  pdf.setDrawColor(...colors.cyan);
  pdf.setLineWidth(0.5);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 4;

  pdf.setTextColor(...colors.cyan);
  pdf.setFontSize(14);
  pdf.setFont('helvetica', 'bold');
  pdf.text(selectedTract.name || 'Tract', margin, y + 4);

  pdf.setTextColor(...riskColor);
  pdf.setFontSize(9);
  pdf.setFont('helvetica', 'bold');
  pdf.text(`Risk Level: ${riskLevel}`, pageWidth - margin - 45, y + 4);
  y += tractHeaderH;

  // Tract metrics (2-column grid)
  const col1X = margin;
  const col2X = pageWidth / 2 + 5;

  pdf.setFontSize(8.5);
  pdf.setFont('helvetica', 'normal');
  for (let idx = 0; idx < tractMetrics.length; idx++) {
    const [label, val] = tractMetrics[idx];
    const row = Math.floor(idx / 2);
    const cx = idx % 2 === 0 ? col1X : col2X;
    const cy = y + row * metricLineH;
    pdf.setTextColor(...colors.secondary);
    pdf.text(label + ':', cx, cy);
    pdf.setTextColor(...colors.text);
    pdf.text(val, cx + 32, cy);
  }
  y += tractMetricsH + sectionGap;

  // ==== SCENARIO PARAMETERS ====
  pdf.setDrawColor(...colors.blue);
  pdf.setLineWidth(0.5);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 4;

  pdf.setTextColor(...colors.blue);
  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Scenario Parameters', margin, y + 4);
  y += scenarioHeaderH;

  pdf.setFontSize(8.5);
  pdf.setFont('helvetica', 'normal');
  for (let idx = 0; idx < params_display.length; idx++) {
    const item = params_display[idx];
    const row = Math.floor(idx / 2);
    const cx = idx % 2 === 0 ? col1X : col2X;
    const cy = y + row * metricLineH;
    pdf.setTextColor(...colors.secondary);
    pdf.text(item.label + ':', cx, cy);
    pdf.setTextColor(...colors.emerald);
    pdf.text(item.value, cx + 38, cy);
  }
  y += scenarioMetricsH + sectionGap;

  // ==== RESILIENCE METRICS ====
  pdf.setDrawColor(...colors.blue);
  pdf.setLineWidth(0.5);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 4;

  pdf.setTextColor(...colors.blue);
  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Resilience Forecast', margin, y + 4);
  y += resilienceHeaderH;

  pdf.setFontSize(8.5);
  pdf.setFont('helvetica', 'normal');
  for (let idx = 0; idx < metrics_display.length; idx++) {
    const item = metrics_display[idx];
    const row = Math.floor(idx / 2);
    const cx = idx % 2 === 0 ? col1X : col2X;
    const cy = y + row * metricLineH;
    pdf.setTextColor(...colors.secondary);
    pdf.text(item.label + ':', cx, cy);
    pdf.setTextColor(...item.color);
    pdf.text(item.value, cx + 38, cy);
  }
  y += resilienceMetricsH + sectionGap;

  // ==== IMMEDIATE ACTION ITEMS ====
  pdf.setDrawColor(...colors.blue);
  pdf.setLineWidth(0.5);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 4;

  pdf.setTextColor(...colors.blue);
  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Immediate Action Items', margin, y + 4);
  y += actionSectionHeaderH;

  // Recalculate available space for action items and scale line height to fill it
  const actionAvailableH = footerY - y - 2;
  const totalLines = splitActions.reduce((sum, lines) => sum + lines.length, 0);
  const totalGapsH = (actionItems.length - 1) * actionParaGap;
  const scaledLineH = Math.max(actionLineH, (actionAvailableH - totalGapsH) / Math.max(totalLines, 1));
  // Cap so text doesn't become too spread out if content is short
  const finalLineH = Math.min(scaledLineH, 5.5);

  pdf.setFontSize(actionFontSize);
  pdf.setFont('helvetica', 'normal');

  splitActions.forEach((lines, idx) => {
    // Accent bar before each paragraph
    pdf.setDrawColor(...colors.blue);
    pdf.setLineWidth(0.8);
    pdf.line(margin, y - 0.5, margin + 3, y - 0.5);

    pdf.setTextColor(...colors.text);
    lines.forEach((line) => {
      pdf.text(line, margin + 5, y);
      y += finalLineH;
    });

    if (idx < splitActions.length - 1) y += actionParaGap;
  });

  // ==== FOOTER (page 1) ====
  pdf.setTextColor(...colors.muted);
  pdf.setFontSize(6.5);
  pdf.setFont('helvetica', 'normal');
  pdf.text('WAVE — Workforce Analytics & Vulnerability Engine • Santa Barbara County', margin, footerY + 3);
  pdf.text(`Page 1 of 2 | Generated: ${new Date().toISOString().split('T')[0]}`, pageWidth - margin - 45, footerY + 3);

  // ============================================================
  // PAGE 2 — Policy Recommendations & Budget Allocation
  // ============================================================
  pdf.addPage();

  // Background
  pdf.setFillColor(10, 11, 15);
  pdf.rect(0, 0, pageWidth, pageHeight, 'F');

  // Header bar
  pdf.setFillColor(...colors.blue);
  pdf.rect(0, 0, pageWidth, headerHeight, 'F');
  pdf.setTextColor(232, 232, 240);
  pdf.setFontSize(22);
  pdf.setFont('helvetica', 'bold');
  pdf.text('POLICY RECOMMENDATIONS', margin + 5, 13);
  pdf.setFontSize(9);
  pdf.setFont('helvetica', 'normal');
  pdf.setTextColor(...colors.cyan);
  pdf.text(`Budget Allocation • ${selectedTract.name || 'Tract'} • ${new Date().toLocaleDateString()}`, margin + 5, 22);

  const budget = simData?.emergency_fund || 10_000_000;
  const allocation = generateBudgetAllocation(selectedTract, params, simData);

  // Available height for page 2 content
  const p2Start = headerHeight + 6;
  const p2FooterY = pageHeight - footerHeight;
  const p2Available = p2FooterY - p2Start;

  // Layout constants
  const numCategories = allocation.length;         // 6
  const barSectionH   = p2Available * 0.42;        // top 42% = budget bars
  const policySectionH = p2Available * 0.52;       // bottom 52% = policy text
  const rowH = barSectionH / numCategories;
  const barMaxW = contentWidth * 0.50;
  const maxPct = Math.max(...allocation.map(a => a.pct));

  let p2y = p2Start;

  // --- Budget Allocation title ---
  pdf.setDrawColor(...colors.cyan);
  pdf.setLineWidth(0.5);
  pdf.line(margin, p2y, pageWidth - margin, p2y);
  p2y += 4;
  pdf.setTextColor(...colors.cyan);
  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Relief Budget Allocation', margin, p2y + 4);

  // Total budget badge
  const budgetLabel = budget >= 1_000_000
    ? `Total: $${(budget / 1_000_000).toFixed(1)}M`
    : `Total: $${(budget / 1_000).toFixed(0)}K`;
  pdf.setTextColor(...colors.emerald);
  pdf.setFontSize(9);
  pdf.text(budgetLabel, pageWidth - margin - pdf.getTextWidth(budgetLabel) - 2, p2y + 4);
  p2y += 10;

  // Bar chart rows
  const barColors = [
    colors.blue, colors.cyan, colors.emerald,
    colors.purple, [255, 165, 0], colors.rose
  ];
  const labelColW = contentWidth * 0.34;
  const amountColX = margin + labelColW + barMaxW + 4;

  allocation.forEach((cat, i) => {
    const rowY = p2y + i * rowH;
    const midY = rowY + rowH * 0.55;
    const barH  = rowH * 0.30;
    const barW  = (cat.pct / maxPct) * barMaxW;
    const barX  = margin + labelColW;

    // Category label
    pdf.setTextColor(...colors.text);
    pdf.setFontSize(7.5);
    pdf.setFont('helvetica', 'bold');
    // Truncate label to fit column
    const labelText = pdf.splitTextToSize(cat.name, labelColW - 3)[0];
    pdf.text(labelText, margin, midY);

    // Bar background
    pdf.setFillColor(30, 32, 42);
    pdf.rect(barX, midY - barH, barMaxW, barH, 'F');

    // Bar fill
    pdf.setFillColor(...barColors[i % barColors.length]);
    pdf.rect(barX, midY - barH, barW, barH, 'F');

    // Percentage label on bar
    pdf.setTextColor(232, 232, 240);
    pdf.setFontSize(6.5);
    pdf.setFont('helvetica', 'bold');
    pdf.text(`${(cat.pct * 100).toFixed(1)}%`, barX + 2, midY - 1);

    // Dollar amount
    const amtStr = cat.amount >= 1_000_000
      ? `$${(cat.amount / 1_000_000).toFixed(2)}M`
      : `$${(cat.amount / 1_000).toFixed(0)}K`;
    pdf.setTextColor(...barColors[i % barColors.length]);
    pdf.setFontSize(7.5);
    pdf.setFont('helvetica', 'bold');
    pdf.text(amtStr, amountColX, midY);
  });

  p2y += barSectionH + 4;

  // --- Policy Rationale title ---
  pdf.setDrawColor(...colors.blue);
  pdf.setLineWidth(0.5);
  pdf.line(margin, p2y, pageWidth - margin, p2y);
  p2y += 4;
  pdf.setTextColor(...colors.blue);
  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Policy Rationale', margin, p2y + 4);
  p2y += 10;

  // Pre-split all rationale texts and scale line height to fill remaining space
  const rationaleTexts = allocation.map(cat =>
    pdf.splitTextToSize(policyRationale(cat.name, selectedTract, params, simData), contentWidth - 6)
  );
  const totalRatLines = rationaleTexts.reduce((s, l) => s + l.length, 0);
  const ratParaGap = 2.5;
  const ratAvailH = p2FooterY - p2y - 2 - (numCategories - 1) * ratParaGap;
  const ratLineH = Math.min(4.2, Math.max(3.0, ratAvailH / Math.max(totalRatLines, 1)));

  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(7.8);

  rationaleTexts.forEach((lines, i) => {
    const cat = allocation[i];
    // Colored category label
    pdf.setTextColor(...barColors[i % barColors.length]);
    pdf.setFontSize(7.5);
    pdf.setFont('helvetica', 'bold');
    pdf.text(cat.name + ':', margin, p2y);
    const labelW = pdf.getTextWidth(cat.name + ':') + 3;

    // First line inline with label, rest indented
    pdf.setTextColor(...colors.text);
    pdf.setFontSize(7.8);
    pdf.setFont('helvetica', 'normal');
    lines.forEach((line, li) => {
      const xOff = li === 0 ? margin + labelW : margin + 4;
      const yOff = li === 0 ? p2y : p2y + li * ratLineH;
      pdf.text(line, xOff, yOff);
    });
    p2y += lines.length * ratLineH + ratParaGap;
  });

  // ==== FOOTER (page 2) ====
  pdf.setTextColor(...colors.muted);
  pdf.setFontSize(6.5);
  pdf.setFont('helvetica', 'normal');
  pdf.text('Coastal Labor-Resilience Engine • Santa Barbara County', margin, p2FooterY + 3);
  pdf.text(`Page 2 of 2 | ID: ${Date.now()}`, pageWidth - margin - 30, p2FooterY + 3);

  // Save PDF
  const tractName = (selectedTract.name || 'Tract').replace(/\s+/g, '_');
  const fileName = `Resilience_Report_${tractName}_${new Date().toISOString().split('T')[0]}.pdf`;
  pdf.save(fileName);

  return fileName;
}
