'use client'

import { useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase/client'
import { Staff } from '@/lib/types'

export function useStaff() {
  const [staff, setStaff] = useState<Staff | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const supabase = createClient()

    async function fetchStaff() {
      const {
        data: { user },
      } = await supabase.auth.getUser()

      if (!user) {
        setLoading(false)
        return
      }

      const { data } = await supabase
        .from('staff')
        .select('*')
        .eq('id', user.id)
        .single()

      setStaff(data ?? null)
      setLoading(false)
    }

    fetchStaff()
  }, [])

  return { staff, loading }
}
