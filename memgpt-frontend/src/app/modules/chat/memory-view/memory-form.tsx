import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@memgpt/components/form';
import { Textarea } from '@memgpt/components/textarea';
import { Button } from '@memgpt/components/button';
import React from 'react';
import { AgentMemory } from '../../../libs/agents/agent-memory';
import { cn } from '@memgpt/utils';
import { AgentMemoryUpdateSchema } from '../../../libs/agents/agent-memory-update';
import { useAgentMemoryUpdateMutation } from '../../../libs/agents/use-agent-memory.mutation';
import { Loader2 } from 'lucide-react';
import { useCurrentAgent } from '../../../libs/agents/agent.store';
import { useAgentMemoryQuery } from '../../../libs/agents/use-agent-memory.query';


export function MemoryForm({ className }: { className?: string }) {
  const currentAgent = useCurrentAgent();
  const { data, isLoading } = useAgentMemoryQuery(currentAgent?.name);
  const mutation = useAgentMemoryUpdateMutation();

  const form = useForm<z.infer<typeof AgentMemoryUpdateSchema>>({
    resolver: zodResolver(AgentMemoryUpdateSchema),
    defaultValues: {
      persona: data?.core_memory?.persona,
      human: data?.core_memory?.human,
      user_id: 'null',
      agent_id: currentAgent?.name,
    },
  });

  function onSubmit(data: z.infer<typeof AgentMemoryUpdateSchema>) {
    mutation.mutate(data);
  }

  if (isLoading) return <p>Is loading</p>;

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className={cn('flex flex-col gap-8', className)}>
        <FormField
          control={form.control}
          name="persona"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Persona</FormLabel>
              <FormControl>
                <Textarea
                  className="min-h-[20rem] resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                This is the agents core memory. It is immediately available without querying any other resources.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="human"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Human</FormLabel>
              <FormControl>
                <Textarea
                  className="min-h-[20rem] resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                This is what the agent knows about you so far!
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button className="mt-4" type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : undefined}
          {mutation.isPending ? 'Saving Changes' : 'Save Changes'}</Button>
      </form>
    </Form>
  );
}
