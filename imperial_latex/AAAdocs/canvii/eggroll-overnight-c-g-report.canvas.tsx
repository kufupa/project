import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const runs = [
  {
    id: "C",
    name: "run_c_fresh_pop32_b32_r2_seedbatch3_alpha005_full120",
    job: "2832319.pbs-7",
    pop: 32,
    batch: 32,
    rank: 2,
    alpha: 0.005,
    episodesPerMember: 3,
    totalUpdates: 30,
    trainRollouts: 2880,
    evalRows: 16,
    evalEpisodes: 800,
    bestUpdate: 24,
    bestSuccessEpisodes: 13,
    bestPc: 26,
    avgSumReward: 60.41,
    avgMaxReward: 3.47,
    walltime: "06:14:11",
    walltimeHours: 6.24,
    cpuTime: "60:05:57",
    memGiB: 63.17,
  },
  {
    id: "D",
    name: "run_d_fresh_pop64_b32_r1_seedbatch2_alpha0075_full120",
    job: "2832320.pbs-7",
    pop: 64,
    batch: 32,
    rank: 1,
    alpha: 0.0075,
    episodesPerMember: 2,
    totalUpdates: 28,
    trainRollouts: 3584,
    evalRows: 15,
    evalEpisodes: 750,
    bestUpdate: 12,
    bestSuccessEpisodes: 13,
    bestPc: 26,
    avgSumReward: 67.75,
    avgMaxReward: 3.33,
    walltime: "07:36:05",
    walltimeHours: 7.6,
    cpuTime: "88:36:41",
    memGiB: 58.59,
  },
  {
    id: "E",
    name: "run_e_fresh_pop32_b32_r2_seedbatch4_alpha004_full120",
    job: "2832321.pbs-7",
    pop: 32,
    batch: 32,
    rank: 2,
    alpha: 0.004,
    episodesPerMember: 4,
    totalUpdates: 24,
    trainRollouts: 3072,
    evalRows: 13,
    evalEpisodes: 650,
    bestUpdate: 0,
    bestSuccessEpisodes: 12,
    bestPc: 24,
    avgSumReward: 60.88,
    avgMaxReward: 3.0,
    walltime: "06:24:22",
    walltimeHours: 6.41,
    cpuTime: "64:58:00",
    memGiB: 63.38,
  },
  {
    id: "F",
    name: "run_f_fresh_pop64_b32_r2_seedbatch2_alpha004_full120",
    job: "2832322.pbs-7",
    pop: 64,
    batch: 32,
    rank: 2,
    alpha: 0.004,
    episodesPerMember: 2,
    totalUpdates: 24,
    trainRollouts: 3072,
    evalRows: 13,
    evalEpisodes: 650,
    bestUpdate: 4,
    bestSuccessEpisodes: 12,
    bestPc: 24,
    avgSumReward: 55.28,
    avgMaxReward: 3.06,
    walltime: "06:24:22",
    walltimeHours: 6.41,
    cpuTime: "61:59:04",
    memGiB: 53.87,
  },
  {
    id: "G",
    name: "run_g_fresh_pop128_b32_r1_seedbatch2_alpha0035_full120",
    job: "2832323.pbs-7",
    pop: 128,
    batch: 32,
    rank: 1,
    alpha: 0.0035,
    episodesPerMember: 2,
    totalUpdates: 20,
    trainRollouts: 5120,
    evalRows: 11,
    evalEpisodes: 550,
    bestUpdate: 4,
    bestSuccessEpisodes: 14,
    bestPc: 28,
    avgSumReward: 75.17,
    avgMaxReward: 3.64,
    walltime: "08:58:01",
    walltimeHours: 8.97,
    cpuTime: "113:48:00",
    memGiB: 51.19,
  },
];

const commonUpdates = ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20"];

const commonSuccessSeries = [
  { name: "C: pop32 rank2 alpha0.005", data: [7, 8, 9, 12, 5, 7, 6, 8, 8, 9, 8] },
  { name: "D: pop64 rank1 alpha0.0075", data: [7, 10, 9, 10, 8, 8, 13, 9, 7, 8, 8] },
  { name: "E: pop32 rank2 alpha0.004", data: [12, 6, 9, 10, 5, 11, 3, 8, 5, 8, 5] },
  { name: "F: pop64 rank2 alpha0.004", data: [9, 8, 12, 4, 7, 10, 5, 7, 5, 5, 7] },
  { name: "G: pop128 rank1 alpha0.0035", data: [7, 10, 14, 9, 10, 8, 9, 5, 7, 8, 10] },
];

const bestRun = runs.reduce((best, run) =>
  run.bestSuccessEpisodes > best.bestSuccessEpisodes ? run : best
);

const totalUpdates = runs.reduce((sum, run) => sum + run.totalUpdates, 0);
const totalTrainRollouts = runs.reduce((sum, run) => sum + run.trainRollouts, 0);
const totalEvalEpisodes = runs.reduce((sum, run) => sum + run.evalEpisodes, 0);

function OutcomeSummary() {
  const theme = useHostTheme();
  return (
    <Card size="lg">
      <CardHeader trailing="all clean">
        overnight result
      </CardHeader>
      <CardBody>
        <Grid columns={4} gap={16}>
          <Stat value={`${bestRun.bestSuccessEpisodes}/50`} label={`Best eval episodes (${bestRun.id})`} tone="success" />
          <Stat value="5/5" label="Jobs completed" tone="success" />
          <Stat value={`${totalUpdates}`} label="Total train updates" />
          <Stat value={`${totalTrainRollouts.toLocaleString()}`} label="Train rollouts" />
        </Grid>
        <Divider />
        <Text>
          Best checkpoint was run {bestRun.id} at update {bestRun.bestUpdate}: {bestRun.bestSuccessEpisodes}/50 eval episodes succeeded, avg sum reward {bestRun.avgSumReward}.
        </Text>
        <Text tone="secondary" size="small" style={{ color: theme.text.secondary }}>
          Monitor ended cleanly. No OOM, no traceback, no FileNotFoundError, no batch-16 fallback.
        </Text>
      </CardBody>
    </Card>
  );
}

export default function EggrollOvernightReport() {
  const theme = useHostTheme();

  return (
    <Stack gap={20}>
      <H1>EGGROLL Overnight C-G Report</H1>
      <Text tone="secondary">
        Source: `artifacts/phase50_eggroll_next/20260523_203005` · task `push-v3` · eval `50` episodes per checkpoint · eval parallelism `vector_async`, `n_envs=25`, chunk length `5`.
      </Text>

      <OutcomeSummary />

      <Grid columns="1.2fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Eval Success Curve</H2>
          <Text size="small" tone="secondary">
            X-axis: training update. Y-axis: successful eval episodes out of 50. Common window only: updates 0-20 so all runs align.
          </Text>
          <LineChart
            categories={commonUpdates}
            series={commonSuccessSeries}
            height={260}
            valueSuffix="/50"
          />
          <Text size="small" tone="tertiary">
            Source: `eval_sweep_summary.json` for runs C-G. Later C/D-only points are in table below.
          </Text>
        </Stack>

        <Stack gap={10}>
          <H2>Best Checkpoint</H2>
          <Text size="small" tone="secondary">
            X-axis: run. Y-axis: successful eval episodes out of 50.
          </Text>
          <BarChart
            categories={runs.map((run) => run.id)}
            series={[{ name: "Best eval success episodes", data: runs.map((run) => run.bestSuccessEpisodes), tone: "success" }]}
            height={260}
            valueSuffix="/50"
          />
          <Text size="small" tone="tertiary">
            Source: overnight summaries and eval sweep best rows.
          </Text>
        </Stack>
      </Grid>

      <H2>Run Outcomes</H2>
      <Table
        headers={[
          "Run",
          "Hyperparams",
          "Updates",
          "Train rollouts",
          "Best eval",
          "Best update",
          "Avg sum reward",
          "Walltime",
        ]}
        rows={runs.map((run) => [
          run.id,
          `pop ${run.pop}, batch ${run.batch}, rank ${run.rank}, alpha ${run.alpha}, ep/member ${run.episodesPerMember}`,
          `${run.totalUpdates}`,
          run.trainRollouts.toLocaleString(),
          `${run.bestSuccessEpisodes}/50 (${run.bestPc}%)`,
          `${run.bestUpdate}`,
          run.avgSumReward.toFixed(2),
          run.walltime,
        ])}
        columnAlign={["left", "left", "right", "right", "right", "right", "right", "right"]}
        rowTone={[undefined, undefined, undefined, undefined, "success"]}
        striped
      />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Walltime</H2>
          <Text size="small" tone="secondary">
            X-axis: run. Y-axis: elapsed walltime in hours.
          </Text>
          <BarChart
            categories={runs.map((run) => run.id)}
            series={[{ name: "Walltime", data: runs.map((run) => run.walltimeHours), tone: "info" }]}
            height={230}
            valueSuffix="h"
          />
          <Text size="small" tone="tertiary">
            Source: PBS footer in each `pbs.out`. All jobs requested 18h and finished under 9h.
          </Text>
        </Stack>

        <Card>
          <CardHeader trailing={`${totalEvalEpisodes.toLocaleString()} eval episodes`}>
            resource usage
          </CardHeader>
          <CardBody>
            <Table
              headers={["Run", "CPU time", "Memory used", "PBS job"]}
              rows={runs.map((run) => [
                run.id,
                run.cpuTime,
                `${run.memGiB.toFixed(2)} GiB`,
                run.job,
              ])}
              columnAlign={["left", "right", "right", "left"]}
              framed={false}
            />
            <Text size="small" tone="secondary" style={{ marginTop: 12 }}>
              Requested resources: 1 RTX6000 GPU, 64 CPUs, 192 GB RAM per run. Actual peak memory stayed near 51-63 GiB.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <H2>Readout</H2>
      <Grid columns={3} gap={16}>
        <Card>
          <CardHeader>What worked</CardHeader>
          <CardBody>
            <Text>Batch 32 held stable across all five jobs. Source-copy plus scratch checkpointing prevented live-tree deletion issues.</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>What underperformed</CardHeader>
          <CardBody>
            <Text>Most runs peaked early, then declined or oscillated. E best was base checkpoint, so some training settings harmed eval.</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Next signal</CardHeader>
          <CardBody>
            <Text>G is strongest peak: pop 128, rank 1, alpha 0.0035. Next experiment should test shorter runs around update 4-8, or lower alpha after early gain.</Text>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="info" title="Checkpoint note">
        Best available checkpoint is G update 0004: `run_g_fresh_pop128_b32_r1_seedbatch2_alpha0035_full120/checkpoints/update_0004.pt`.
      </Callout>
    </Stack>
  );
}
