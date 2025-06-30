import os
import json
from typing import AsyncGenerator, Final, List, Literal, Optional, Union, cast
import multiprocessing
from queue import Empty as QueueEmpty
from vllm.sequence import RequestMetrics
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    CompletionResponse,
    ChatCompletionResponse,
    UsageInfo,
)


class MetricsSaverClient:
    def __init__(self, save_dir: Optional[str] = None):
        self.queue = multiprocessing.Queue()
        if save_dir is None:
            save_dir = os.environ.get("VLLM_REQUEST_METRICS_DIR", None)
            if save_dir is None:
                save_dir = os.path.join(os.getcwd(), "request_metrics")
            else:
                save_dir = save_dir.strip('"')
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.process = multiprocessing.Process(
            target=_save_metrics_worker, args=(self.queue, self.save_dir)
        )
        self.process.start()
        print("Metrics saver process started")

    def save_request_metrics(self, request_metrics, request_ids):
        """Add metrics to the queue for processing"""
        self.queue.put(("metrics", request_metrics, request_ids))

    def save_usage(self, usage: UsageInfo):
        """Add usage to the queue for processing"""
        self.queue.put(("usage", usage))

    def save_metadata(self, metadata: dict):
        """Add metadata to the queue for processing"""
        self.queue.put(("metadata", metadata))

    def save_response(self, response: CompletionResponse):
        """Add response to the queue for processing"""
        self.queue.put(("response", response))

    def save_chat_response(self, response: ChatCompletionResponse):
        """Add chatting response to the queue for processing"""
        self.queue.put(("chat_response", response))

    def save_request(self, request: CompletionRequest):
        """Add request to the queue for processing"""
        self.queue.put(("request", request))

    def save_chat_request(
        self, request: ChatCompletionRequest, request_id: Optional[str] = None
    ):
        """Add chatting request to the queue for processing"""
        self.queue.put(("chat_request", request, request_id))

    def shutdown(self):
        """Clean shutdown of the metrics saver"""
        self.queue.put(None)  # Send sentinel value
        self.process.join()


def _save_metrics_worker(queue: multiprocessing.Queue, save_dir: Optional[str] = None):
    """Worker process that saves metrics from the queue"""
    saver = MetricsSaver(save_dir=save_dir)
    while True:
        try:
            # Timeout allows the process to check for termination
            item = queue.get()
            if item is None:  # Sentinel value to stop the process
                break
            if item[0] == "metrics":
                request_metrics, request_ids = item[1:]
                saver.save_request_batch_metrics(request_metrics, request_ids)
                saver.save_request_metrics(request_metrics, request_ids)
                saver.save_request_metrics_stats(request_metrics, request_ids)
            elif item[0] == "request":
                saver.save_request(item[1])
            elif item[0] == "chat_request":
                saver.save_chat_request(item[1], item[2] if len(item) > 2 else None)
            elif item[0] == "usage":
                saver.save_usage(item[1])
            elif item[0] == "response":
                saver.save_response(item[1])
            elif item[0] == "chat_response":
                saver.save_chat_response(item[1])
            elif item[0] == "metadata":
                saver.save_metadata(item[1])
            else:
                raise ValueError(f"Unknown item type: {item[0]}")
        except QueueEmpty:
            print("Queue is empty, waiting for new metrics", flush=True)
            continue
        except Exception as e:
            print(f"Error saving metrics: {e}", flush=True)


class MetricsSaver:
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir

    def save_request(self, request: CompletionRequest):
        save_path = os.path.join(self.save_dir, "llm_request.txt")
        with open(save_path, "a+") as f:
            f.write(f"{request.model_dump_json()}\n")

    def save_chat_request(
        self, request: ChatCompletionRequest, request_id: Optional[str] = None
    ):
        if request_id is not None:
            try:
                request.request_id = request_id
            except AttributeError:
                # If request does not have request_id, we can skip setting it
                pass
        save_path = os.path.join(self.save_dir, "llm_chat_request.txt")
        with open(save_path, "a+") as f:
            f.write(f"{request.model_dump_json()}\n")

    def save_response(self, response: CompletionResponse):
        save_path = os.path.join(self.save_dir, "llm_response.txt")
        with open(save_path, "a+") as f:
            f.write(f"{response.model_dump_json()}\n")

    def save_chat_response(self, response: ChatCompletionResponse):
        save_path = os.path.join(self.save_dir, "llm_chat_response.txt")
        with open(save_path, "a+") as f:
            f.write(f"{response.model_dump_json()}\n")

    def save_usage(self, usage: UsageInfo):
        save_path = os.path.join(self.save_dir, "llm_usage.txt")
        with open(save_path, "a+") as f:
            f.write(f"{usage.model_dump_json()}\n")

    def save_metadata(self, metadata: dict):
        save_path = os.path.join(self.save_dir, "llm_metadata.txt")
        with open(save_path, "a+") as f:
            f.write(f"{json.dumps(metadata)}\n")

    def save_request_metrics(
        self, request_metrics: List[RequestMetrics], request_ids: List[str]
    ):
        """
        Save the request metrics for each request in the batch in csv format
        This function is used for debugging and performance analysis.
        """
        assert len(request_metrics) == len(
            request_ids
        ), "request_metrics and request_ids must have the same length"
        request_metrics_cols = [
            "embd_id",
            "request_id",
            "arrival_time",
            "last_token_time",
            "first_scheduled_time",
            "time_in_queue",
            "finished_time",
            "scheduler_time",
            "model_forward_time",
            "model_execute_time",
        ]
        save_path = os.path.join(self.save_dir, "request_metrics.csv")
        if not os.path.exists(save_path):
            # Create the file and write the header
            with open(save_path, "w+") as f:
                f.write(",".join(request_metrics_cols) + "\n")

        # Save the metrics for each request
        with open(save_path, "a+") as f:
            for i in range(len(request_metrics)):
                metric = request_metrics[i]
                request_id = request_ids[i]
                embd_id = "-".join(request_id.split("-")[:-1])
                row = [
                    embd_id,
                    request_id,
                    str(metric.arrival_time),
                    str(metric.last_token_time),
                    str(metric.first_scheduled_time),
                    str(metric.time_in_queue),
                    str(metric.finished_time),
                    str(metric.scheduler_time),
                    str(metric.model_forward_time),
                    str(metric.model_execute_time),
                ]
                f.write(",".join(row) + "\n")

    def save_request_batch_metrics(
        self, request_metrics: List[RequestMetrics], request_ids: List[str]
    ):
        assert len(request_metrics) == len(
            request_ids
        ), "request_metrics and request_ids must have the same length"
        batch_metrics_cols = [
            "embd_id",
            "batch_id",
            "batch_size",
            "min_request_id",
            "max_request_id",
            "min_arrival_time",
            "max_arrival_time",
            "first_scheduled_time",
            "last_token_time",
            "finished_time",
            "arrival_cost",
            "scheduler_cost",
            "compute_cost",
            "total_cost",
        ]
        save_path = os.path.join(self.save_dir, "request_batch_metrics.csv")
        if not os.path.exists(save_path):
            # Create the file and write the header
            with open(save_path, "w+") as f:
                f.write(",".join(batch_metrics_cols) + "\n")

        metrics_group = {}
        request_ids_group = {}
        for i, metric in enumerate(request_metrics):
            first_scheduled_time = metric.first_scheduled_time
            if first_scheduled_time not in metrics_group:
                metrics_group[first_scheduled_time] = []
                request_ids_group[first_scheduled_time] = []
            metrics_group[first_scheduled_time].append(metric)
            request_ids_group[first_scheduled_time].append(request_ids[i])
        # Save the metrics for each batch
        with open(save_path, "a+") as f:
            for batch_id, key in enumerate(sorted(metrics_group.keys())):
                metrics = metrics_group[key]
                request_ids = request_ids_group[key]
                batch_size = len(metrics)
                embd_id = "-".join(request_ids[0].split("-")[:-1])
                min_request_id = min(request_ids, key=lambda x: int(x.split("-")[-1]))
                max_request_id = max(request_ids, key=lambda x: int(x.split("-")[-1]))

                arrival_times = [m.arrival_time for m in metrics]
                min_arrival_time = min(arrival_times)
                max_arrival_time = max(arrival_times)
                arrival_cost = max_arrival_time - min_arrival_time
                first_scheduled_time = metrics[0].first_scheduled_time
                last_token_time = metrics[0].last_token_time
                finished_time = min([m.finished_time for m in metrics])
                scheduler_cost = metrics[0].scheduler_time
                compute_cost = finished_time - first_scheduled_time
                total_cost = max([m.finished_time for m in metrics]) - min_arrival_time
                row = [
                    embd_id,
                    str(batch_id),
                    str(batch_size),
                    str(min_request_id),
                    str(max_request_id),
                    str(min_arrival_time),
                    str(max_arrival_time),
                    str(first_scheduled_time),
                    str(last_token_time),
                    str(finished_time),
                    str(arrival_cost),
                    str(scheduler_cost),
                    str(compute_cost),
                    str(total_cost),
                ]
                f.write(",".join(row) + "\n")
                # print(f"{batch_id}, {min_request_id}: {row}")

    def save_request_metrics_stats(
        self, request_metrics: List[RequestMetrics], request_ids: List[str]
    ):
        """
        Save the statistics of request metrics in csv format
        This function is used for debugging and performance analysis.
        """
        assert len(request_metrics) == len(
            request_ids
        ), "request_metrics and request_ids must have the same length"
        request_metrics_cols = [
            "embd_id",
            "batch_size",
            "num_steps",
            "min_arrival_time",
            "max_arrival_time",
            "min_finished_time",
            "max_finished_time",
            "arrival_cost",
            "compute_cost",
            "schedule_cost",
            "total_cost",
        ]
        save_path = os.path.join(self.save_dir, "request_metrics_stats.csv")
        if not os.path.exists(save_path):
            # Create the file and write the header
            with open(save_path, "w+") as f:
                f.write(",".join(request_metrics_cols) + "\n")

        embed_id = "-".join(request_ids[0].split("-")[:-1])
        batch_size = len(request_metrics)
        min_arrival_time = min([m.arrival_time for m in request_metrics])
        max_arrival_time = max([m.arrival_time for m in request_metrics])
        min_finished_time = min([m.finished_time for m in request_metrics])
        max_finished_time = max([m.finished_time for m in request_metrics])
        arrival_cost = max_arrival_time - min_arrival_time
        total_cost = max_finished_time - min_arrival_time

        schedule_cost = 0.0
        compute_cost = 0.0
        num_steps = 0
        metrics_group = {}
        request_ids_group = {}
        for i, metric in enumerate(request_metrics):
            first_scheduled_time = metric.first_scheduled_time
            if first_scheduled_time not in metrics_group:
                metrics_group[first_scheduled_time] = []
                request_ids_group[first_scheduled_time] = []
                compute_cost += metric.finished_time - first_scheduled_time
                schedule_cost += metric.scheduler_time
                num_steps += 1
            metrics_group[first_scheduled_time].append(metric)
            request_ids_group[first_scheduled_time].append(request_ids[i])

        # Save the metrics for each request
        with open(save_path, "a+") as f:
            row = [
                embed_id,
                str(batch_size),
                str(num_steps),
                str(min_arrival_time),
                str(max_arrival_time),
                str(min_finished_time),
                str(max_finished_time),
                str(arrival_cost),
                str(compute_cost),
                str(schedule_cost),
                str(total_cost),
            ]
            f.write(",".join(row) + "\n")
            # print(f"{embed_id}: {row}")
