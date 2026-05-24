import { createClient } from '@/lib/supabase/server'
import { Staff } from '@/lib/types'

export async function getStaff(): Promise<Staff | null> {
  const supabase = await createClient()

  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) return null

  const { data } = await supabase
    .from('staff')
    .select('*')
    .eq('id', user.id)
    .single()

  return data ?? null
}
